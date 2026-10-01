// The price of D-3s's forming (`d3s-v0-f16`, form_sm120.cu) on sm_120a, per element per side, in rate loops from registers
// (rates.py): the steps that read the unit's stream (rate_salt) and the steps that read the data alone (rate_free), each in
// the instruction mix form_rows issues them in, and three one-instruction controls.  Each thread runs G groups of four
// entries (two f16x2 words), independent of each other; a group's step per repeat is
//
//   rate_salt  c = fma(1/4, t, -3/8), t = 0x3C00 | (w & 0x3FF);  e = A * c;  x = the other group's e (form_rows' gather of
//              e o pi, here a register);  s = e + x;  [Q(h + e) | Q(h + x) | Q((-h) + (-s))], Q the e4m3x2 cast, the three
//              4-code words b0, b1, b2 (the negations folded into the add);  next w = (b0 ^ b2, b1), the loop's one extra instruction a group
//   rate_free  h = dec(codes), p = the other group's h (the partner);  A = max(max(|h|, |p|), floor);  z = h * 2^-8;
//              q = fma(z, z, q);  next codes = codes ^ A0 ^ A1, the loop's one extra instruction a group
//
// Args (form_sm120.cu's): p0 the inputs (u32, threads x G x 8), p1 the outputs (u32, threads x G x 8: the last repeat's
// b0, b1, b2 and w, or A0, A1, q0, q1 and codes); i0 repeats, i1 threads.  The controls chain eight independent registers
// a thread through one instruction each a repeat: FFMA, HFMA2, LOP3, HADD2, HMUL2, HMNMX2, the e4m3x2 cast and its decode
// (two F2FPs, encdec), the decode alone (F2FP.UNPACK_B) or PRMT.
#include <cstdint>

#define DEV __device__ __forceinline__

struct Args {
  uint64_t p[16];
  int i[16];
};

constexpr int G = 2;

DEV uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm volatile("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEV uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm volatile("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEV uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
DEV uint32_t hmax2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm volatile("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// (-a) + (-b): the negations are operand modifiers (one HFMA2 -a, 1, -b), so the cheapest form issues one instruction
DEV uint32_t hnadd2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm volatile("{.reg .b32 x, y; neg.f16x2 x, %1; neg.f16x2 y, %2; add.rn.f16x2 %0, x, y;}" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// max(max(|a|, |b|), c): |.| is HMNMX2's operand modifier; sm_120's PTX has no three-input f16x2 max
DEV uint32_t amp2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("{.reg .b32 x, y, z; abs.f16x2 x, %1; abs.f16x2 y, %2; max.f16x2 z, x, y; max.f16x2 %0, z, %3;}"
               : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
DEV uint32_t dec2(uint32_t two) {
  uint32_t d;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"((uint16_t)(two & 0xFFFF)));
  return d;
}
DEV uint32_t enc2(uint32_t x) {
  uint16_t d;
  asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(x));
  return d;
}
DEV uint32_t enc4(uint32_t lo, uint32_t hi) { return enc2(lo) | enc2(hi) << 16; }
DEV uint32_t centre2(uint32_t w) { return hfma2(0x34003400u, (w & 0x03FF03FFu) | 0x3C003C00u, 0xB600B600u); }

extern "C" __global__ void __launch_bounds__(256) rate_salt(const Args a) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, reps = a.i[0];
  if (tid >= a.i[1]) return;
  const uint32_t* in = reinterpret_cast<const uint32_t*>(a.p[0]) + (size_t)tid * G * 8;
  uint32_t h[G][2], A[G][2], w[G][2], b[G][3];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int j = 0; j < 2; ++j) h[g][j] = in[8 * g + j], A[g][j] = in[8 * g + 2 + j], w[g][j] = in[8 * g + 4 + j];
#pragma unroll 4
  for (int r = 0; r < reps; ++r) {
    uint32_t e[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int j = 0; j < 2; ++j) e[g][j] = hmul2(A[g][j], centre2(w[g][j]));
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const uint32_t* x = e[(g + 1) % G];
      const uint32_t s0 = hadd2(e[g][0], x[0]), s1 = hadd2(e[g][1], x[1]);
      b[g][0] = enc4(hadd2(h[g][0], e[g][0]), hadd2(h[g][1], e[g][1]));
      b[g][1] = enc4(hadd2(h[g][0], x[0]), hadd2(h[g][1], x[1]));
      b[g][2] = enc4(hnadd2(h[g][0], s0), hnadd2(h[g][1], s1));
      w[g][0] = b[g][0] ^ b[g][2];
      w[g][1] = b[g][1];
    }
  }
  uint32_t* out = reinterpret_cast<uint32_t*>(a.p[1]) + (size_t)tid * G * 8;
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int j = 0; j < 3; ++j) out[8 * g + j] = b[g][j];
    out[8 * g + 3] = w[g][0];
    out[8 * g + 4] = w[g][1];
  }
}

extern "C" __global__ void __launch_bounds__(256) rate_free(const Args a) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, reps = a.i[0];
  if (tid >= a.i[1]) return;
  const uint32_t* in = reinterpret_cast<const uint32_t*>(a.p[0]) + (size_t)tid * G * 8;
  uint32_t c[G], q[G][2], A[G][2];
  const uint32_t fl = in[7];
#pragma unroll
  for (int g = 0; g < G; ++g) c[g] = in[8 * g], q[g][0] = 0, q[g][1] = 0;
#pragma unroll 4
  for (int r = 0; r < reps; ++r) {
    uint32_t h[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g) h[g][0] = dec2(c[g]), h[g][1] = dec2(c[g] >> 16);
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const uint32_t* p = h[(g + 1) % G];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        A[g][j] = amp2(h[g][j], p[j], fl);
        const uint32_t z = hmul2(h[g][j], 0x1C001C00u);
        q[g][j] = hfma2(z, z, q[g][j]);
      }
      c[g] ^= A[g][0] ^ A[g][1];
    }
  }
  uint32_t* out = reinterpret_cast<uint32_t*>(a.p[1]) + (size_t)tid * G * 8;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    out[8 * g] = A[g][0], out[8 * g + 1] = A[g][1], out[8 * g + 2] = q[g][0], out[8 * g + 3] = q[g][1];
    out[8 * g + 4] = c[g];
  }
}

template <int OP>
DEV void control(const Args& a) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, reps = a.i[0];
  if (tid >= a.i[1]) return;
  const uint32_t* in = reinterpret_cast<const uint32_t*>(a.p[0]) + (size_t)tid * G * 8;
  uint32_t v[8];
  const uint32_t k = in[8], m = in[9];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = in[j];
#pragma unroll 4
  for (int r = 0; r < reps; ++r)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (OP == 0) {
        float f = __uint_as_float(v[j]);
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(__uint_as_float(k)), "f"(__uint_as_float(m)));
        v[j] = __float_as_uint(f);
      } else if (OP == 1) {
        v[j] = hfma2(v[j], k, m);
      } else if (OP == 2) {
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v[j]) : "r"(k), "r"(m));
      } else if (OP == 3) {
        v[j] = hadd2(v[j], k);
      } else if (OP == 4) {
        v[j] = hmul2(v[j], k);
      } else if (OP == 5) {
        v[j] = hmax2(v[j], k);
      } else if (OP == 6) {
        v[j] = dec2(enc2(v[j]));
      } else if (OP == 7) {
        v[j] = dec2(v[j]);
      } else {
        asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(v[j]) : "r"(k));
      }
    }
  uint32_t* out = reinterpret_cast<uint32_t*>(a.p[1]) + (size_t)tid * G * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) out[j] = v[j];
}
extern "C" __global__ void __launch_bounds__(256) rate_ffma(const Args a) { control<0>(a); }
extern "C" __global__ void __launch_bounds__(256) rate_hfma2(const Args a) { control<1>(a); }
extern "C" __global__ void __launch_bounds__(256) rate_lop3(const Args a) { control<2>(a); }
extern "C" __global__ void __launch_bounds__(256) rate_hadd2(const Args a) { control<3>(a); }
extern "C" __global__ void __launch_bounds__(256) rate_hmul2(const Args a) { control<4>(a); }
extern "C" __global__ void __launch_bounds__(256) rate_hmnmx2(const Args a) { control<5>(a); }
extern "C" __global__ void __launch_bounds__(256) rate_encdec(const Args a) { control<6>(a); }
extern "C" __global__ void __launch_bounds__(256) rate_dec(const Args a) { control<7>(a); }
extern "C" __global__ void __launch_bounds__(256) rate_prmt(const Args a) { control<8>(a); }
