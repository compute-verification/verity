// The frozen decoder's non-matmul units on sm_90a and sm_120a, each word its twin's (words.py and the unit modules' OPS
// docstrings): RMSNorm, the per-row s8 quantizer (units_quant up to 16384 words, units_quant4 a cluster of four CTAs for
// 65536), softmax over a QK^T row, argmax over the head, ReLU, the residual add, and units_sweep, which runs the helpers
// below on every operand word so check.py can hold them to the twins' instructions exhaustively.
//
// Every f32 operation is a named IEEE round-to-nearest-even instruction (__fadd_rn, __fsub_rn, __fmul_rn, __fdiv_rn,
// __fsqrt_rn, __fmaf_rn), built with -fmad=false and no FTZ flag, so nothing is contracted or flushed; the one approximate
// operation is ex2.approx.ftz.f32 (MUFU.EX2, `mufu.MufuEx2Ftz`), in softmax only.  The sweep also runs MUFU.RSQ's two PTX
// forms, which no unit reads, and only in the sm_120a cubin (SM120 below).  Every f32 reduction is sequential in a
// fixed order from +0 by one thread; threads meet only through shared memory, shuffles, __syncthreads and the cluster
// barrier.  No atomics.  Only the integer maxima (the quantizer's absmax tree) are taken in a free order: an integer max is
// exact, so every node's word is the same in any order.
//
// Rows move as uint4s, coalesced: RMSNorm and softmax stage a CTA's rows in shared memory, swizzled so that both the
// coalesced copy and one thread's walk down its own chunk or row meet distinct bank groups; the quantizer keeps its row
// in registers.
//
// Every division's operands are runtime values (the constants 1 and 127 come in as kernel arguments): ptxas folds a
// constant numerator into a copy of div.rn's slow path that the harness SASS gate's pins don't cover.  No kernel divides
// integers (the host passes the counts), so the integer kernels hold no floating-point instruction at all.
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define D __device__ __forceinline__
#define FULL 0xFFFFFFFFu
// What only the sm_120a cubin holds: the sm_90a cubin stays byte for byte the one the H200's check ran
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define SM120 1
#else
#define SM120 0
#endif

D float fadd(float a, float b) { return __fadd_rn(a, b); }
D float fsub(float a, float b) { return __fsub_rn(a, b); }
D float fmul(float a, float b) { return __fmul_rn(a, b); }
D float fdiv(float a, float b) { return __fdiv_rn(a, b); }
D float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
D float fsqrt(float a) { return __fsqrt_rn(a); }
D float f32(u32 w) { return __uint_as_float(w); }
D float widen(u32 h) { return __uint_as_float((h & 0xFFFFu) << 16); }
D u16 bf16(float f) { u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
D float ex2(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
D u32 s8(float x) { int r; asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(r) : "f"(x)); return (u32)r & 0xFFu; }
#if SM120
D float rsq_ftz(float x) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
D float rsq(float x) { float r; asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
#endif
D int order16(u32 v) { int m = (int)(v & 0x7FFFu); return (v & 0x8000u) ? -m : m; }
D bool isnan16(u32 v) { return (v & 0x7F80u) == 0x7F80u && (v & 0x7Fu); }
// +0 where 0 > v strictly and v is not NaN: exactly the words 0x8001..0xFF80
D u32 relu16(u32 v) { v &= 0xFFFFu; return v - 0x8001u < 0x7F80u ? 0u : v; }
D u32 relu2(u32 w) { return relu16(w) | (relu16(w >> 16) << 16); }
// RN(amax / 127) when 1 <= f32(amax) <= +inf as signed words, else 1.0; f127 is 127.0's word
D u32 quant_scale(u32 amax, u32 f127) {
  const u32 a = (amax & 0xFFFFu) << 16;
  return ((int)a >= 1 && (int)a <= 0x7F800000) ? __float_as_uint(fdiv(f32(a), f32(f127))) : 0x3F800000u;
}

// ---- the quantizer's quotient RN(f32(h) / s), s the row's scale (positive: normal, subnormal, +inf or 1.0) -------------
// y = RN(1 / s) once a row; then q0 = RN(f32(h) y) and q = RN(q0 + RN-exact(f32(h) - s q0) y), two FMAs (Markstein's
// correction).  That is RN(f32(h) / s) whenever s's exponent is in [2, 251] and h's in [max(32, e_s - 100), min(254, e_s +
// 100)]; +-0 is itself; every other word (subnormal, inf, NaN, or a quotient near the range's ends) takes div.rn.  The
// sweep (op 5) holds qdiv to RN(f32(h) / scale(amax)) on every bf16 word h and every amax.
struct QRow {
  float s, y;
  u32 lo, span;
};

D QRow qrow(u32 s_word, u32 one) {
  QRow r;
  r.s = f32(s_word);
  r.y = fdiv(f32(one), r.s);
  const u32 e = (s_word >> 23) & 0xFFu;
  const bool ok = e >= 2u && e <= 251u;
  r.lo = ok ? (e > 132u ? e - 100u : 32u) : 256u;
  r.span = ok ? min(e + 100u, 254u) - r.lo : 0u;
  return r;
}

D bool qslow(u32 h, const QRow& r) { return ((h >> 7) & 0xFFu) - r.lo > r.span && (h & 0x7FFFu); }

D float qfast(u32 h, const QRow& r) {
  const float a = widen(h);
  const float q0 = fmul(a, r.y);
  const float q = ffma(ffma(-r.s, q0, a), r.y, q0);
  return (h & 0x7FFFu) ? q : a;
}

D float qdiv(u32 h, const QRow& r) { return qslow(h, r) ? fdiv(widen(h), r.s) : qfast(h, r); }

// ---- RMSNorm: one CTA of RT threads per row, N <= 16384 = nc C, nc = ng G, C % 8 == 0, nc <= RT ---------------------------
#define RT 128
#define RQ 2048
// chunk c's piece k (16 uint4s a chunk at C = 128) at (c << 4) | (k ^ (c & 15))
D int swz(int q, bool sw) { return sw ? q ^ ((q >> 4) & 15) : q; }

D u32 rms_pair(u32 x2, u32 w2, float rs) {
  const u32 lo = bf16(fmul(widen(w2), widen(bf16(fmul(widen(x2), rs)))));
  const u32 hi = bf16(fmul(widen(w2 >> 16), widen(bf16(fmul(widen(x2 >> 16), rs)))));
  return lo | (hi << 16);
}

extern "C" __global__ void __launch_bounds__(RT) units_rmsnorm(const u16* __restrict__ x, const u16* __restrict__ w,
                                                               u16* __restrict__ out, u32* __restrict__ chunks,
                                                               u32* __restrict__ groups, u32* __restrict__ rs_out, int N,
                                                               int C, int nc, int G, int ng, u32 n_word, u32 eps_word,
                                                               u32 one_word) {
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
    for (int k = 0; k < ng; ++k) total = fadd(total, gs[k]);
    const float mean = fdiv(total, f32(n_word));
    const float rs = fdiv(f32(one_word), fsqrt(fadd(mean, f32(eps_word))));
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
}

// ---- the s8 quantizer.  A CTA of RT threads holds P = N / S words of a row in registers, thread t the uint4s t + RT j.
// The absmax tree's level 1 takes chunks of min(128, N) words of the (relu'd) words with their sign bits cleared: in load
// j, chunk (t >> 4) + 8 j is half-warp t >> 4's, so a half-warp's shuffle max makes it.  The CTA's node is the max of its
// level-1 words: the root at S = 1, and level 2's word at S = 4 (N = 65536: 512 -> 4 -> 1), whose root the cluster's four
// nodes make through distributed shared memory.  tree gets every level's words in order, the root last.  Then each
// thread's codes, every word by qdiv: qfast in one pass, and a thread with any word outside qfast's window recomputes its
// words by qdiv from x.  N % 8 == 0; N <= 16384 (every width >= 128 a multiple of 128) at S = 1, N = 65536 at S = 4 -----
#define QJ (RQ / RT)

D u32 amax8(const uint4& v) {
  const u32 m = 0x7FFFu;
  return max(max(max(v.x & m, v.x >> 16 & m), max(v.y & m, v.y >> 16 & m)),
             max(max(v.z & m, v.z >> 16 & m), max(v.w & m, v.w >> 16 & m)));
}

D u32 qcode2(u32 w, const QRow& r) { return s8(qfast(w & 0xFFFFu, r)) | (s8(qfast(w >> 16, r)) << 8); }
D bool qslow2(u32 w, const QRow& r) { return qslow(w & 0xFFFFu, r) | qslow(w >> 16, r); }

template <int S>
D void quant_cta(const u16* __restrict__ x, uint8_t* __restrict__ codes, u32* __restrict__ scale_out,
                 u16* __restrict__ tree, int N, int T, int relu, u32 f127, u32 one, size_t row, int rank) {
  __shared__ u32 l1[RT];
  __shared__ u32 node_s, root_s;
  const int t = threadIdx.x, P = N / S, p8 = P >> 3, nc = N >= 128 ? P >> 7 : 1;
  const size_t base = row * (size_t)N + (size_t)rank * P;
  u16* tr = tree ? tree + row * (size_t)T : nullptr;
  const uint4* xp = reinterpret_cast<const uint4*>(x + base);
  uint4 v[QJ];
#pragma unroll
  for (int j = 0; j < QJ; ++j) {
    const int q = t + RT * j;
    v[j] = q < p8 ? xp[q] : make_uint4(0u, 0u, 0u, 0u);
  }
  if (relu) {
#pragma unroll
    for (int j = 0; j < QJ; ++j) v[j] = make_uint4(relu2(v[j].x), relu2(v[j].y), relu2(v[j].z), relu2(v[j].w));
  }
#pragma unroll
  for (int j = 0; j < QJ; ++j) {
    u32 m = amax8(v[j]);
#pragma unroll
    for (int o = 8; o; o >>= 1) m = max(m, __shfl_xor_sync(FULL, m, o));
    const int c = (t >> 4) + 8 * j;
    if ((t & 15) == 0 && c < nc) {
      l1[c] = m;
      if (tr) tr[rank * nc + c] = (u16)m;
    }
  }
  __syncthreads();
  if (t < 32) {
    u32 m = 0;
    for (int c = t; c < nc; c += 32) m = max(m, l1[c]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(FULL, m, o));
    if (t == 0) {
      node_s = m;
      if (S == 1) {
        root_s = m;
        if (tr && nc > 1) tr[nc] = (u16)m;
      } else if (tr) {
        tr[S * nc + rank] = (u16)m;
      }
    }
  }
  if (S > 1) {
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    if (t == 0) {
      u32 m = 0;
      for (int k = 0; k < S; ++k) m = max(m, *cl.map_shared_rank(&node_s, k));
      root_s = m;
      if (tr && rank == 0) tr[S * nc + S] = (u16)m;
    }
    cl.sync();
  } else {
    __syncthreads();
  }
  const u32 sw = quant_scale(root_s, f127);
  if (t == 0 && rank == 0 && scale_out) scale_out[row] = sw;
  const QRow qr = qrow(sw, one);
  uint2* cp = reinterpret_cast<uint2*>(codes + base);
  bool slow = false;
#pragma unroll
  for (int j = 0; j < QJ; ++j) {
    const int q = t + RT * j;
    if (q < p8) {
      cp[q] = make_uint2(qcode2(v[j].x, qr) | (qcode2(v[j].y, qr) << 16), qcode2(v[j].z, qr) | (qcode2(v[j].w, qr) << 16));
      slow = slow | qslow2(v[j].x, qr) | qslow2(v[j].y, qr) | qslow2(v[j].z, qr) | qslow2(v[j].w, qr);
    }
  }
  if (slow) {
#pragma unroll 1
    for (int e = 0; e < QJ * 8; ++e) {
      const int q = t + RT * (e >> 3);
      if (q < p8) {
        const size_t i = 8 * (size_t)q + (e & 7);
        const u32 h = relu ? relu16(x[base + i]) : (u32)x[base + i];
        codes[base + i] = (uint8_t)s8(qdiv(h, qr));
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(RT) units_quant(const u16* __restrict__ x, uint8_t* __restrict__ codes,
                                                             u32* __restrict__ scale_out, u16* __restrict__ tree, int N,
                                                             int T, int relu, u32 f127, u32 one) {
  quant_cta<1>(x, codes, scale_out, tree, N, T, relu, f127, one, blockIdx.x, 0);
}

extern "C" __global__ void __cluster_dims__(4, 1, 1) __launch_bounds__(RT)
    units_quant4(const u16* __restrict__ x, uint8_t* __restrict__ codes, u32* __restrict__ scale_out,
                 u16* __restrict__ tree, int N, int T, int relu, u32 f127, u32 one) {
  quant_cta<4>(x, codes, scale_out, tree, N, T, relu, f127, one, blockIdx.x >> 2, (int)cg::this_cluster().block_rank());
}

// ---- softmax over QK^T rows of LMAX = 32 words: a CTA of SR threads takes rows heads h0 + t of sequence b = blockIdx.y
// (h0 = SR blockIdx.x) at n = lens[b] keys (clamped to 1..32), one thread a row.  The rows come in and go out as uint4s
// through shared memory, row r's piece k at (r << 2) | (k ^ ((r >> 1) & 3)); the words past n are read and never used ----
#define LMAX 32
#define SR 128
D int sswz(int q) { return q ^ ((q >> 3) & 3); }

extern "C" __global__ void __launch_bounds__(SR) units_softmax(const u16* __restrict__ x, const int* __restrict__ lens,
                                                               u16* __restrict__ p, u16* __restrict__ m_out,
                                                               u32* __restrict__ r_out, int heads, u32 one_word) {
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
    const float r = fdiv(f32(one_word), s);
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

// ---- ReLU on bf16 words: n / 8 uint4s, the n % 8 words past them one at a time ------------------------------------------
extern "C" __global__ void units_relu(const u16* __restrict__ x, u16* __restrict__ y, u64 n) {
  const uint4* xp = reinterpret_cast<const uint4*>(x);
  uint4* yp = reinterpret_cast<uint4*>(y);
  const u64 stride = (u64)gridDim.x * blockDim.x, n8 = n >> 3;
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
    const uint4 v = xp[i];
    yp[i] = make_uint4(relu2(v.x), relu2(v.y), relu2(v.z), relu2(v.w));
  }
  for (u64 i = (n8 << 3) + (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) y[i] = (u16)relu16(x[i]);
}

// ---- the residual add, Bf16AddF2fp_v1 per element: bf16(RN(widen(a) + widen(b))) -----------------------------------------
D u32 add_pair(u32 a2, u32 b2) {
  return bf16(fadd(widen(a2), widen(b2))) | ((u32)bf16(fadd(widen(a2 >> 16), widen(b2 >> 16))) << 16);
}

// n8 = n / 8 uint4s, the n % 8 words past them one at a time
extern "C" __global__ void units_add(const u16* __restrict__ a, const u16* __restrict__ b, u16* __restrict__ y, u64 n) {
  const uint4* ap = reinterpret_cast<const uint4*>(a);
  const uint4* bp = reinterpret_cast<const uint4*>(b);
  uint4* yp = reinterpret_cast<uint4*>(y);
  const u64 stride = (u64)gridDim.x * blockDim.x, n8 = n >> 3;
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
    const uint4 u = ap[i], v = bp[i];
    yp[i] = make_uint4(add_pair(u.x, v.x), add_pair(u.y, v.y), add_pair(u.z, v.z), add_pair(u.w, v.w));
  }
  for (u64 i = (n8 << 3) + (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    y[i] = bf16(fadd(widen(a[i]), widen(b[i])));
}

// ---- the sweep: CTA b of 1024 threads covers chunk b, the operands i in [b 2^24, (b + 1) 2^24) below n, and writes
// sum_i splitmix64((i << 32) | op(i)) mod 2^64 (sweep.py) --------------------------------------------------------------
D u64 splitmix64(u64 z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

D u32 sweep_op(int op, u32 i, u32 arg, u32 one) {
  switch (op) {
    case 0: return __float_as_uint(ex2(f32(i)));
    case 1: return s8(f32(i));
    case 2: return __float_as_uint(fsqrt(f32(i)));
    case 3: return __float_as_uint(fdiv(f32(arg), f32(i)));
    case 4: return __float_as_uint(fdiv(f32(i), f32(arg)));
    case 5: return __float_as_uint(qdiv(i & 0xFFFFu, qrow(quant_scale(i >> 16, arg), one)));
    case 6: return bf16(f32(i));
    case 7: return quant_scale(i, arg);
    case 8: return bf16(fadd(widen(i >> 16), widen(i)));
#if SM120
    case 9: return __float_as_uint(rsq_ftz(f32(i)));
    case 10: return __float_as_uint(rsq(f32(i)));
#endif
    default: return 0xFFFFFFFFu;
  }
}

extern "C" __global__ void __launch_bounds__(1024) units_sweep(int op, u64 n, u32 arg, u32 one, u64* __restrict__ partial) {
  __shared__ u64 ws[32];
  const u64 base = (u64)blockIdx.x << 24;
  u64 acc = 0;
  for (u32 k = 0; k < (1u << 14); ++k) {
    const u64 i = base | ((u64)k << 10) | threadIdx.x;
    if (i >= n) break;
    acc += splitmix64((i << 32) | sweep_op(op, (u32)i, arg, one));
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

#if SM120
// op(i) for the operands i in [lo, lo + count): a chunk whose digest disagrees, word by word
extern "C" __global__ void units_sweep_words(int op, u64 lo, u32 count, u32 arg, u32 one, u32* __restrict__ out) {
  const u32 k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < count) out[k] = sweep_op(op, (u32)(lo + k), arg, one);
}
#endif
