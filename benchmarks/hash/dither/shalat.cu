// One SHA-512 compression's latency on one SM, the A table's critical path (h1c_a_call2_k is 27 of them in sequence at
// K 4096): each lane runs N compressions, each one's state the next one's input, its message independent of the state (as
// a segment's bytes are), and lane 0 of each warp reads clock64 around the chain.  One block a launch.  Forms:
//   0  sh::sha512_compress, the schedule inline (h1c's hb_block, form 0's compression), on `warps` warps
//   1  h1c.cu's ap_cmp and ap_serve as h1c_a_call2_k runs them: consumer warps 0 and 1, producer warps 2 and 3
//   2  form 1 with the consumer's half inlined (no call: the message and state stay in registers)
//   3  form 2 with the round reassociated: e' = (d + h + kw) + S1(e) + Ch, a' = (h + kw) + S1(e) + Ch + S0(a) + Maj,
//      the sums of h, kw and d off the e chain
//   4  form 3 with the off-chain sums (h + kw, d + h + kw) on the FMA pipe (IMAD with an opaque one)
// Bit-exactness: every form's final states equal form 0's (shalat.py compares them).
#include "h1c.cu"

namespace lat {
__device__ __forceinline__ u64 S1(u64 e) { return sh::rotr64(e, 14) ^ sh::rotr64(e, 18) ^ sh::rotr64(e, 41); }
__device__ __forceinline__ u64 S0(u64 a) { return sh::rotr64(a, 28) ^ sh::rotr64(a, 34) ^ sh::rotr64(a, 39); }
__device__ __forceinline__ u64 add_fma(u64 x, u64 y, u32 one) {  // x + y on IMAD.WIDE and IMAD (the FMA pipe)
  u64 r;
  asm("{\n\t.reg .u32 xl, xh, rl, rh;\n\t"
      "mov.b64 {xl, xh}, %1;\n\t"
      "mad.wide.u32 %0, xl, %3, %2;\n\t"
      "mov.b64 {rl, rh}, %0;\n\t"
      "mad.lo.u32 rh, xh, %3, rh;\n\t"
      "mov.b64 %0, {rl, rh};\n\t}"
      : "=l"(r)
      : "l"(x), "l"(y), "r"(one));
  return r;
}
template <int F>
__device__ __forceinline__ void rnd(u64& a, u64& b, u64& c, u64& d, u64& e, u64& f, u64& g, u64& h, u64 kw, u32 one) {
  if constexpr (F <= 2) {
    ap_round(a, b, c, d, e, f, g, h, kw);
  } else {
    const u64 hk = F == 4 ? add_fma(h, kw, one) : h + kw;
    const u64 dhk = F == 4 ? add_fma(d, hk, one) : d + hk;
    const u64 s1 = S1(e), ch = (e & f) ^ (~e & g);
    const u64 s0 = S0(a), mj = (a & b) ^ (a & c) ^ (b & c);
    const u64 en = dhk + s1 + ch;
    const u64 an = (hk + s1 + ch) + (s0 + mj);
    h = g;
    g = f;
    f = e;
    e = en;
    d = c;
    c = b;
    b = a;
    a = an;
  }
}
// ap_cmp inlined, with round form F
template <int F>
__device__ __forceinline__ void cmp_in(u64 hs[8], const u64 w[16], u32 one) {
  const int lane = threadIdx.x & 31, p = threadIdx.x >> 5, bar = 1 + 5 * p;
  u64* base = ap_base(p);
  u64* x = base + lane;
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    x[32 * (80 + t)] = w[t];
    x[32 * t] = sh::kSha512K[t] + w[t];
  }
  if (lane == 0) base[96 * 32] = 0;
  ap_arrive(bar, 64);
  u64 a = hs[0], b = hs[1], c = hs[2], d = hs[3], e = hs[4], f = hs[5], g = hs[6], h = hs[7];
#pragma unroll 1
  for (int q = 0; q < 5; ++q) {
    if (q) ap_sync(bar + q, 64);
    u64 kw[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) kw[i] = x[32 * (16 * q + i)];
#pragma unroll
    for (int i = 0; i < 16; ++i) rnd<F>(a, b, c, d, e, f, g, h, kw[i], one);
  }
  hs[0] += a;
  hs[1] += b;
  hs[2] += c;
  hs[3] += d;
  hs[4] += e;
  hs[5] += f;
  hs[6] += g;
  hs[7] += h;
}
__device__ __forceinline__ void message(u64 w[16], int lane, int it) {
#pragma unroll
  for (int t = 0; t < 16; ++t) w[t] = 0x9e3779b97f4a7c15ull * (u64)(1 + t + 16 * it) ^ (u64)lane << 40;
}
}  // namespace lat

// out: per warp, its cycles (clock64) and ns (%globaltimer) over the chain; states: per thread of a consumer warp, its
// final eight words
template <int F>
__global__ void __launch_bounds__(128) shalat_k(int n, u32 one, long long* cyc, long long* ns, u64* states) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (F >= 1 && wid >= 2) {
    ap_serve(wid - 2);
    return;
  }
  u64 hs[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) hs[i] = sh::kSha512Iv[i] ^ (u64)lane;
  __syncwarp();
  long long c0 = clock64();
  u64 t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  for (int it = 0; it < n; ++it) {
    u64 w[16];
    lat::message(w, lane, it);
    if constexpr (F == 0) {
      sh::sha512_compress(hs, w);
    } else if constexpr (F == 1) {
      u64 t[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) t[i] = hs[i];
      ap_cmp(t, w);
#pragma unroll
      for (int i = 0; i < 8; ++i) hs[i] = t[i];
    } else {
      lat::cmp_in<F>(hs, w, one);
    }
  }
  __syncwarp();
  long long c1 = clock64();
  u64 t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  if (F >= 1) ap_stop();
  if (lane == 0) {
    cyc[wid] = c1 - c0;
    ns[wid] = (long long)(t1 - t0);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) states[(wid * 32 + lane) * 8 + i] = hs[i];
}

extern "C" int shalat(int form, int warps, int n, long long* cyc, long long* ns, u64* states) {
  const u32 one = 1;
  switch (form) {
    case 0:
      shalat_k<0><<<1, 32 * warps>>>(n, one, cyc, ns, states);
      break;
#define SHALAT_PAIRED(F)                                                                              \
  case F:                                                                                             \
    if (cudaFuncSetAttribute(shalat_k<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, AP_SMEM)) return 3; \
    shalat_k<F><<<1, 128, AP_SMEM>>>(n, one, cyc, ns, states);                                       \
    break;
      SHALAT_PAIRED(1)
      SHALAT_PAIRED(2)
      SHALAT_PAIRED(3)
      SHALAT_PAIRED(4)
#undef SHALAT_PAIRED
    default:
      return 2;
  }
  return (int)cudaGetLastError();
}
