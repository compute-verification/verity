// D-3s's forming (`d3s-v0-f16`, #295's `D3Family("d3s", "xq", sums="f16")`) on sm_120a: one side's stacked rows from its E4M3
// codes, the noise stream and NCP's permutation.  Per row of k codes (an activation row, or a weight column):
//
//   h_l    = the code's f16 value;  floor = max(f16(rms), 2^-6),  rms = sqrt(ss / k) * 256 in f32, ss over 64 lanes (lane
//            l mod 64 chains fma.f16 of (h_l * 2^-8)^2 over runs of 256 of its entries, widening each run to f32 and adding
//            it), then an f32 butterfly over the lanes (32 ... 1)
//   A_l    = max(|h_l|, |h_partner(l)|, floor), partner = pi^-1 (A rows) or pi (B rows)
//   e_l    = A_l * fma(1/4, t_l, -3/8),  t_l = 0x3C00 | (w_l & 0x3FF), w_l the stream's l-th little-endian 16-bit word
//   A:  e2 = e o pi,       s = e + e2;   row = [Q(h + e) | Q(h + e2) | Q((-h) + (-s))]
//   B:  f2 = -(e o pi^-1), s = e + f2;   row = [Q(h + e) | Q(h + f2) | Q(h + s)]
//
// every operation one f16 rounding (add.rn, mul.rn, fma.rn on half2), Q = cvt.rn.satfinite.e4m3x2.f16x2.  pi is word-granular
// (four entries a word, their order kept), so both gathers read whole 4-entry words.
//
// The stream is 2k bytes a row.  XOF_SHAKE: SHAKE256 of the row's message, as #295 specifies (`xof`): its squeeze is
// sequential, so form_c_shake runs a thread a row and writes the centred values c_l = fma(1/4, t_l, -3/8) to global memory
// for form_rows.  XOF_B3: BLAKE3's XOF of the same message, whose output blocks are independent compressions (counter i):
// form_rows computes them in the row's CTA, a thread a 64-byte block, and nothing goes through global memory.  The host
// absorbs each row's message (SHAKE: the padded rate blocks; BLAKE3: the root node's chaining value and last block); the
// message is the same for every row but its last frame, the row index, so on the device it is one or two blocks of
// absorption against the stream's k / 68 (SHAKE) or k / 32 (BLAKE3) output permutations.
//
// Args (pearl_c_sm120's: p[16], i[16]):
//   form_c_shake  p0 the rows' padded message lanes (u64, rows x nb x 17), p1 c (u16, rows x k); i0 rows, i1 k, i2 nb
//   form_c_shake2 form_c_shake's, and i3 1 to write the raw stream (2k bytes a row) into p1 instead of c
//   form_rows_*   p0 codes (u8, rows x k), p1 c (u16, rows x k; read by _g), p2 B3 roots (u32, rows x 32: cv[8], block[16],
//                 len, flags; read by _b3), p3 pi's words (u16, k / 4), p4 pi^-1's words, p5 out (u8, rows x pitch); i0
//                 rows, i1 k, i2 left, i3 pitch.  One CTA a row, FORM_THREADS threads, dynamic shared memory 3k + 256 bytes.
//   xof_raw       p0 SHAKE lanes, p1 B3 roots, p2 out (u8, rows x n); i0 rows, i1 n, i2 nb, i3 0 SHAKE / 1 BLAKE3
//   gen_any       p0 out; i0, i2 count (low, high), i1 seed: every finite E4M3 code (NaN codes 0x7F, 0xFF become 0x7E, 0xFE)
#include <cstdint>

#include "../pouw_hash/pouw_hash.cuh"

#define DEV __device__ __forceinline__

struct Args {
  uint64_t p[16];
  int i[16];
};

constexpr int FORM_THREADS = 256;
constexpr int SHAKE_RATE = 17;   // SHAKE256's rate in 64-bit lanes (136 bytes)

// f16x2 words (the low half the lower index), each operation one PTX instruction with its rounding written out
DEV uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEV uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEV uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
DEV uint32_t hmax2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEV uint32_t hneg2(uint32_t a) { return a ^ 0x80008000u; }
DEV uint16_t hmul(uint16_t a, uint16_t b) {
  uint16_t d;
  asm("mul.rn.f16 %0, %1, %2;" : "=h"(d) : "h"(a), "h"(b));
  return d;
}
DEV uint16_t hfma(uint16_t a, uint16_t b, uint16_t c) {
  uint16_t d;
  asm("fma.rn.f16 %0, %1, %2, %3;" : "=h"(d) : "h"(a), "h"(b), "h"(c));
  return d;
}
DEV uint16_t hmax(uint16_t a, uint16_t b) {
  uint16_t d;
  asm("max.f16 %0, %1, %2;" : "=h"(d) : "h"(a), "h"(b));
  return d;
}
DEV float h2f(uint16_t a) {
  float d;
  asm("cvt.f32.f16 %0, %1;" : "=f"(d) : "h"(a));
  return d;
}
DEV uint16_t f2h(float a) {
  uint16_t d;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(d) : "f"(a));
  return d;
}
// two codes (the low byte the low half) to f16x2, exactly
DEV uint32_t dec2(uint32_t two) {
  uint32_t d;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"((uint16_t)(two & 0xFFFF)));
  return d;
}
// f16x2 to two codes (the low half the low byte), RN-even, saturating to +-448
DEV uint32_t enc2(uint32_t x) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(x));
  return d;
}
DEV uint32_t enc4(uint32_t lo, uint32_t hi) { return enc2(lo) | enc2(hi) << 16; }

// two stream words (the low half the lower l) to their centred values fma(1/4, t, -3/8)
DEV uint32_t centre2(uint32_t w) { return hfma2(0x34003400u, (w & 0x03FF03FFu) | 0x3C003C00u, 0xB600B600u); }

// One BLAKE3 output block: the compression's 16 words (the XOF's block `counter` of a root node).
DEV void b3_out16(const uint32_t cv[8], const uint32_t msg[16], uint64_t counter, uint32_t len, uint32_t flags,
                  uint32_t out[16]) {
  const uint32_t IV[4] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A};
  const int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t v[16], m[16], t[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = cv[i];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[8 + i] = IV[i];
  v[12] = (uint32_t)counter;
  v[13] = (uint32_t)(counter >> 32);
  v[14] = len;
  v[15] = flags;
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = msg[i];
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    ph::b3_g(v, 0, 4, 8, 12, m[0], m[1]);
    ph::b3_g(v, 1, 5, 9, 13, m[2], m[3]);
    ph::b3_g(v, 2, 6, 10, 14, m[4], m[5]);
    ph::b3_g(v, 3, 7, 11, 15, m[6], m[7]);
    ph::b3_g(v, 0, 5, 10, 15, m[8], m[9]);
    ph::b3_g(v, 1, 6, 11, 12, m[10], m[11]);
    ph::b3_g(v, 2, 7, 8, 13, m[12], m[13]);
    ph::b3_g(v, 3, 4, 9, 14, m[14], m[15]);
#pragma unroll
    for (int i = 0; i < 16; ++i) t[i] = m[PERM[i]];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = t[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = v[i] ^ v[i + 8];
    out[i + 8] = v[i + 8] ^ cv[i];
  }
}

DEV void b3_block(const uint32_t* root, int i, uint32_t out[16]) {
  uint32_t cv[8], msg[16];
#pragma unroll
  for (int j = 0; j < 8; ++j) cv[j] = root[j];
#pragma unroll
  for (int j = 0; j < 16; ++j) msg[j] = root[8 + j];
  b3_out16(cv, msg, (uint64_t)i, root[24], root[25], out);
}

DEV void shake_absorb(const uint64_t* lanes, int nb, uint64_t s[25]) {
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
  for (int b = 0; b < nb; ++b) {
#pragma unroll
    for (int w = 0; w < SHAKE_RATE; ++w) s[w] ^= lanes[SHAKE_RATE * b + w];
    ph::keccak_p<24>(s);
  }
}

extern "C" __global__ void form_c_shake(const Args a) {
  const int rows = a.i[0], k = a.i[1], nb = a.i[2];
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  uint64_t s[25];
  shake_absorb(reinterpret_cast<const uint64_t*>(a.p[0]) + (size_t)r * nb * SHAKE_RATE, nb, s);
  uint2* c = reinterpret_cast<uint2*>(a.p[1]) + (size_t)r * (k / 4);
  const int lanes = k / 4;   // 4 stream words a lane
  for (int L = 0;; L += SHAKE_RATE) {
#pragma unroll
    for (int i = 0; i < SHAKE_RATE; ++i)
      if (L + i < lanes) c[L + i] = make_uint2(centre2((uint32_t)s[i]), centre2((uint32_t)(s[i] >> 32)));
    if (L + SHAKE_RATE >= lanes) break;
    ph::keccak_p<24>(s);
  }
}

// SHAKE256 with a row's state over two threads in Keccak's bit-interleaved form: thread 2r + h holds bits h, h + 2, ... of
// each of row r's 25 lanes in one word.  A 64-bit rotation by 2q rotates each half by q; by 2q + 1 it swaps them (the even
// half becomes the odd one rotated by q + 1, the odd half the even one rotated by q), the swap a shuffle with the other
// thread.  Each thread runs half of each round's instructions, so a row's squeeze, sequential and latency-bound in one
// thread, takes about half as long.
DEV uint32_t rot32(uint32_t x, int n) { return __funnelshift_l(x, x, n); }
DEV uint32_t other(uint32_t x) { return __shfl_xor_sync(0xFFFFFFFFu, x, 1); }
// the perfect shuffle of a word's halves (bit i of the low half to bit 2i, of the high half to 2i + 1), and its inverse
DEV uint32_t zip32(uint32_t x) {
  uint32_t t;
  t = (x ^ (x >> 8)) & 0x0000FF00u, x ^= t ^ (t << 8);
  t = (x ^ (x >> 4)) & 0x00F000F0u, x ^= t ^ (t << 4);
  t = (x ^ (x >> 2)) & 0x0C0C0C0Cu, x ^= t ^ (t << 2);
  t = (x ^ (x >> 1)) & 0x22222222u, x ^= t ^ (t << 1);
  return x;
}
DEV uint32_t unzip32(uint32_t x) {
  uint32_t t;
  t = (x ^ (x >> 1)) & 0x22222222u, x ^= t ^ (t << 1);
  t = (x ^ (x >> 2)) & 0x0C0C0C0Cu, x ^= t ^ (t << 2);
  t = (x ^ (x >> 4)) & 0x00F000F0u, x ^= t ^ (t << 4);
  t = (x ^ (x >> 8)) & 0x0000FF00u, x ^= t ^ (t << 8);
  return x;
}

// Keccak-f[1600] on half h of the state; both threads of the pair call it together
DEV void keccak_f2(uint32_t s[25], int h) {
  const uint32_t RCE[24] = {1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0};
  const uint32_t RCO[24] = {0x00000000u, 0x00000089u, 0x8000008bu, 0x80008080u, 0x0000008bu, 0x00008000u,
                            0x80008088u, 0x80000082u, 0x0000000bu, 0x0000000au, 0x00008082u, 0x00008003u,
                            0x0000808bu, 0x8000000bu, 0x8000008au, 0x80000081u, 0x80000081u, 0x80000008u,
                            0x00000083u, 0x80008003u, 0x80008088u, 0x80000088u, 0x00008000u, 0x80008082u};
  const int ROT[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
  const int PI[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
  const int nh = 1 - h;
#pragma unroll
  for (int r = 0; r < 24; ++r) {
    uint32_t c[5], cp[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) c[i] = ph::lop3<0x96>(ph::lop3<0x96>(s[i], s[i + 5], s[i + 10]), s[i + 15], s[i + 20]);
#pragma unroll
    for (int i = 0; i < 5; ++i) cp[i] = other(c[i]);
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const uint32_t p = c[(i + 4) % 5], q = rot32(cp[(i + 1) % 5], nh);
#pragma unroll
      for (int j = 0; j < 25; j += 5) s[j + i] = ph::lop3<0x96>(s[j + i], p, q);
    }
    uint32_t t = s[1];
#pragma unroll
    for (int i = 0; i < 24; ++i) {
      const int j = PI[i], R = ROT[i];
      const uint32_t x = s[j];
      s[j] = R & 1 ? rot32(other(t), (R >> 1) + nh) : rot32(t, R >> 1);
      t = x;
    }
#pragma unroll
    for (int j = 0; j < 25; j += 5) {
      const uint32_t b0 = s[j], b1 = s[j + 1], b2 = s[j + 2], b3 = s[j + 3], b4 = s[j + 4];
      s[j] = ph::lop3<0xD2>(b0, b1, b2);
      s[j + 1] = ph::lop3<0xD2>(b1, b2, b3);
      s[j + 2] = ph::lop3<0xD2>(b2, b3, b4);
      s[j + 3] = ph::lop3<0xD2>(b3, b4, b0);
      s[j + 4] = ph::lop3<0xD2>(b4, b0, b1);
    }
    s[0] ^= h ? RCO[r] : RCE[r];
  }
}

// form_c_shake's output from two threads a row (blocks of 32 threads, 16 rows); i3 = 1 writes the raw stream instead of c
extern "C" __global__ void __launch_bounds__(32) form_c_shake2(const Args a) {
  const int rows = a.i[0], k = a.i[1], nb = a.i[2], raw = a.i[3];
  const int t = blockIdx.x * blockDim.x + threadIdx.x, h = t & 1;
  const bool live = (t >> 1) < rows;
  const int r = live ? t >> 1 : rows - 1;   // every thread runs to the end: the pair shuffles with the whole warp's mask
  const uint64_t* lanes = reinterpret_cast<const uint64_t*>(a.p[0]) + (size_t)r * nb * SHAKE_RATE;
  uint32_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
  for (int b = 0; b < nb; ++b) {
#pragma unroll
    for (int w = 0; w < SHAKE_RATE; ++w) {
      const uint64_t v = lanes[SHAKE_RATE * b + w];
      s[w] ^= __byte_perm(unzip32((uint32_t)v), unzip32((uint32_t)(v >> 32)), h ? 0x7632 : 0x5410);
    }
    keccak_f2(s, h);
  }
  uint32_t* c = reinterpret_cast<uint32_t*>(a.p[1]) + (size_t)r * (k / 2);
  const int nlanes = k / 4;
  for (int L = 0;; L += SHAKE_RATE) {
#pragma unroll
    for (int i = 0; i < SHAKE_RATE; ++i) {
      const uint32_t w = s[i], pw = other(w);
      // the lane's low word (stream words 4L, 4L + 1) from thread 0, its high word from thread 1
      const uint32_t x = zip32(h ? __byte_perm(pw, w, 0x7632) : __byte_perm(w, pw, 0x5410));
      if (live && L + i < nlanes) c[2 * (L + i) + h] = raw ? x : centre2(x);
    }
    if (L + SHAKE_RATE >= nlanes) break;
    keccak_f2(s, h);
  }
}

template <bool B3>
DEV void form_rows(const Args& a) {
  const int rows = a.i[0], k = a.i[1], left = a.i[2], pitch = a.i[3];
  const int r = blockIdx.x, tid = threadIdx.x;
  if (r >= rows) return;
  extern __shared__ uint32_t smem[];
  uint32_t* sh = smem;                  // the row's codes, 4 a word (k / 4 words)
  uint32_t* se = smem + k / 4;          // c, then e: 2 f16 a word (k / 2 words)
  float* lane = reinterpret_cast<float*>(se + k / 2);   // 64
  const uint32_t* codes = reinterpret_cast<const uint32_t*>(a.p[0]) + (size_t)r * (k / 4);
  for (int w = tid; w < k / 4; w += FORM_THREADS) sh[w] = codes[w];
  if (B3) {
    const uint32_t* root = reinterpret_cast<const uint32_t*>(a.p[2]) + (size_t)r * 32;
    for (int i = tid; i < k / 32; i += FORM_THREADS) {
      uint32_t o[16];
      b3_block(root, i, o);
#pragma unroll
      for (int j = 0; j < 16; ++j) se[16 * i + j] = centre2(o[j]);
    }
  } else {
    const uint32_t* c = reinterpret_cast<const uint32_t*>(a.p[1]) + (size_t)r * (k / 2);
    for (int w = tid; w < k / 2; w += FORM_THREADS) se[w] = c[w];
  }
  __syncthreads();
  // the statistic: lane t chains entries t, t + 64, ... in runs of 256
  if (tid < 64) {
    const uint8_t* hb = reinterpret_cast<const uint8_t*>(sh);
    float acc = 0.f;
    uint16_t q = 0;
    int run = 0;
    for (int l = tid; l < k; l += 64) {
      const uint16_t z = hmul((uint16_t)dec2(hb[l]), 0x1C00);   // h * 2^-8
      q = hfma(z, z, q);
      if (++run == 256) {
        acc = __fadd_rn(acc, h2f(q));
        q = 0;
        run = 0;
      }
    }
    if (run) acc = __fadd_rn(acc, h2f(q));
    lane[tid] = acc;
  }
  __syncthreads();
  for (int half = 32; half; half >>= 1) {
    if (tid < half) lane[tid] = __fadd_rn(lane[tid], lane[tid + half]);
    __syncthreads();
  }
  const float rms = __fmul_rn(__fsqrt_rn(__fdiv_rn(lane[0], (float)k)), 256.f);
  const uint32_t fl = hmax(f2h(rms), 0x2400) * 0x10001u;   // max(f16(rms), 2^-6) in both halves
  const uint16_t* pw = reinterpret_cast<const uint16_t*>(a.p[3]);
  const uint16_t* pinvw = reinterpret_cast<const uint16_t*>(a.p[4]);
  const uint16_t* partner = left ? pinvw : pw;
  // e = A * c, in place
  for (int w = tid; w < k / 4; w += FORM_THREADS) {
    const uint32_t hc = sh[w], pc = sh[partner[w]];
    const uint32_t A0 = hmax2(hmax2(dec2(hc) & 0x7FFF7FFFu, dec2(pc) & 0x7FFF7FFFu), fl);
    const uint32_t A1 = hmax2(hmax2(dec2(hc >> 16) & 0x7FFF7FFFu, dec2(pc >> 16) & 0x7FFF7FFFu), fl);
    se[2 * w] = hmul2(A0, se[2 * w]);
    se[2 * w + 1] = hmul2(A1, se[2 * w + 1]);
  }
  __syncthreads();
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[5]) + (size_t)r * pitch;
  const uint16_t* gather = left ? pw : pinvw;
  for (int w = tid; w < k / 4; w += FORM_THREADS) {
    const uint32_t hc = sh[w];
    const uint32_t h0 = dec2(hc), h1 = dec2(hc >> 16), e0 = se[2 * w], e1 = se[2 * w + 1];
    const int g = gather[w];
    const uint32_t x0 = left ? se[2 * g] : hneg2(se[2 * g]), x1 = left ? se[2 * g + 1] : hneg2(se[2 * g + 1]);
    const uint32_t s0 = hadd2(e0, x0), s1 = hadd2(e1, x1);
    const uint32_t b0 = enc4(hadd2(h0, e0), hadd2(h1, e1));
    const uint32_t b1 = enc4(hadd2(h0, x0), hadd2(h1, x1));
    const uint32_t b2 = left ? enc4(hadd2(hneg2(h0), hneg2(s0)), hadd2(hneg2(h1), hneg2(s1)))
                             : enc4(hadd2(h0, s0), hadd2(h1, s1));
    reinterpret_cast<uint32_t*>(out)[w] = b0;
    reinterpret_cast<uint32_t*>(out + k)[w] = b1;
    reinterpret_cast<uint32_t*>(out + 2 * k)[w] = b2;
  }
}

extern "C" __global__ void __launch_bounds__(FORM_THREADS) form_rows_g(const Args a) { form_rows<false>(a); }
extern "C" __global__ void __launch_bounds__(FORM_THREADS) form_rows_b3(const Args a) { form_rows<true>(a); }

extern "C" __global__ void xof_raw(const Args a) {
  const int rows = a.i[0], n = a.i[1], nb = a.i[2], b3 = a.i[3];
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[2]) + (size_t)r * n;
  if (b3) {
    const uint32_t* root = reinterpret_cast<const uint32_t*>(a.p[1]) + (size_t)r * 32;
    for (int i = 0; 64 * i < n; ++i) {
      uint32_t o[16];
      b3_block(root, i, o);
      for (int j = 0; j < 64 && 64 * i + j < n; ++j) out[64 * i + j] = (uint8_t)(o[j / 4] >> (8 * (j % 4)));
    }
    return;
  }
  uint64_t s[25];
  shake_absorb(reinterpret_cast<const uint64_t*>(a.p[0]) + (size_t)r * nb * SHAKE_RATE, nb, s);
  for (int o = 0;;) {
    for (int w = 0; w < SHAKE_RATE; ++w)
      for (int b = 0; b < 8; ++b, ++o)
        if (o < n) out[o] = (uint8_t)(s[w] >> (8 * b));
    if (o >= n) break;
    ph::keccak_p<24>(s);
  }
}

DEV uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
extern "C" __global__ void gen_any(const Args a) {
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[0]);
  const uint64_t n = (uint64_t)(uint32_t)a.i[0] | (uint64_t)(uint32_t)a.i[2] << 32;
  const uint32_t seed = (uint32_t)a.i[1];
  for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (uint64_t)gridDim.x * blockDim.x) {
    const uint32_t c = mix32((uint32_t)idx * 0x9E3779B9u + seed) & 0xFF;
    out[idx] = (uint8_t)((c & 0x7F) == 0x7F ? c ^ 1 : c);
  }
}
