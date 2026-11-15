// Dither-C's linear (co32c, `checked_strip.co32c_words`) on sm_120a, on the
// honest Dither-1S kernel's mainloop (egemm.cu's e_l7qs: duo every 32 steps, absorbs 7 MMAs behind, consumers asleep on
// the stage barrier).
//
// A linear Y = X W^T (X m x k activations, W n x k weights in [-64, 62]) served as co32c's checked product. Both
// operands are dithered: X = clamp(floor(x / s), -127, 126) + b (egemm.cu's quantizer, b a fresh bit an entry) and
// W^ = W + c (c a fresh bit an entry, so W^ in [-64, 63]). The weights carry U(j, r) = -64 + sum_p w_p beta_p(j, r),
// w = (1, 2, 4, 8, 16, 32, 65), seven fresh coin planes (U in {-64..-1, 1..64}): W' = W^ + U Id^T, int8. Y's residue
// sums S(i, r) = sum_t X(i, 32 t + r) are split into redundant base-128 digits with fresh carry coins z
// (checked_strip.digits): A_d = (R_d mod 128) - 128 z_d, R_{d+1} = floor(R_d / 128) + z_d, A_{m-1} = R_{m-1} (z_{m-2}
// taken as 0 where floor(R_{m-2} / 128) = 127), m = 2 for k <= 4096 and 3 above. Every cell commits, mod 2^32:
//   T_t = sum_{k' < 32 (t + 1)} X W'^T (t < k / 32), O' = T_last - S0 U^T (S0 = A_0), D_d = A_d U^T (1 <= d < m),
// each folded by the duo chains in that order (egemm.cu's e_* chains: derive.py's tile_receipts over T_1 .. T_last,
// then O', then D_1 .. D_{m-1} as more steps); the served output is O = O' - sum_d 128^d D_d = X W^T, exact in s32.
// Noise from the per-call key (sha_pass.cuh's Sig; SHA-512(key || BE64(tag << 56 | j))): b as egemm.cu's (tag 7); c in
// the same patch layout over W (tag 8: block p = (row / 4) (k / 128) + col / 128, entry e = (row % 4) 128 + col % 128,
// bit e % 8 of digest byte e / 8); beta under tag 10 (block j = rows 2j, 2j + 1, entry (row % 2) 32 + r digest byte v,
// beta_p = bit p of v, bit 7 unread); z under tag 9 (block j = rows 4j .. 4j + 3, z_d(i, r) = bit 32 d + r of digest
// bytes 16 (i % 4) .. 16 (i % 4) + 15 read little-endian).
// cohalf (QK^T, k = 128 = the head dimension, r = k / 2 = 64; checked_strip's honest_words at r = 64): K is the served
// side at 7 bits, X = clamp(floor(x / 2s), -64, 62) + b, with Y(r) = X(r) + X(r + 64); Q the peeled side, W' = W^ + U
// over r = col % 64 (U from tag 11: row i's digest byte r, bits 0..6); words T_0 .. T_3, then O = T_last - U Y.
// Kernels:
//   cquant           the dithered quantizer with the tail rows P_A = [A_0 | A_1 | A_2 | 0] (A_2 = 0 at m = 2)
//   cwedit           W' = W + c + U and P_B = [-U | U | U | 0]
//   c_*              the GEMM: e_l7qs's mainloop over X and W', then the tail stage: O' = IMMA (A_0, -U) seeded by T_last,
//                    D_d = IMMA (A_d, U), each word absorbed as it is made, O = O' - 128^d D_d (one IMAD a word a digit);
//                    c_*_o writes O' instead of O (the recombination's cost, and O' for the CPU reference); c_l7qs_v
//                    writes P.V's served value f = float(2 O + 127 S) s_row s_col instead (that epilogue's cost a cell);
//                    c_fx1 is c_l7qs with fx1's fold (egemm.cu's fx_chain<FX1>, one IMAD by
//                    key_mul(1, (4 t + e) 32) on word e of tile position t before duo, f = 4) on every word it absorbs
//   hquant, hwedit   cohalf's X and P_A = [Y_lo | Y_hi | 0 | 0], W' and P_B = [-U_lo | -U_hi | 0 | 0]
//   h_l7qs, h_l2q    cohalf's GEMM (htail: O = T_last - U Y, absorbed), on c_*'s schedules
// Modes (at K = 128, cohalf's kernels):
//   cgemm gate DIR M N K SEED       poisoned buffers: each c_* kernel's O against cuBLASLt's X W^T (W^ = W' - U on the
//                                   device), c_l2q's receipts (another schedule) against c_l7qs's, c_l7qs_v's f against
//                                   vref's, c_fx1's receipts unequal to c_l7qs's, a no-write control; three tiles'
//                                   operands, O, O' and receipts (c_l7qs's and c_fx1's) to DIR for cgemm_ref.py
//   cgemm bench REPS M N K          hot items interleaved across CG_ARMS (egemm.cu's arms too), one JSON line
//   cgemm one ARM M N K             three warm-up calls, then one between cudaProfilerStart and Stop (for ncu)
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a cgemm.cu -lcuda -lcublasLt -lnvidia-ml
// (CGEMM_LIB defined: the kernels without main, as cpv.cu includes them)
#define EGEMM_LIB
#include "egemm.cu"

constexpr uint32_t C_TAG = 8, Z_TAG = 9, BETA_TAG = 10;

// four bits to four int8 bytes 0/1
DEV uint32_t bits01(uint32_t nib) { return (nib * 0x00204081u) & 0x01010101u; }
// U from a digest byte: -64 + sum_p w_p beta_p, w = (1, 2, 4, 8, 16, 32, 65)
__host__ __device__ inline int u_beta(uint32_t v) { return -64 + (int)(v & 63u) + ((v & 64u) ? 65 : 0); }
// H1's narrow U (co32c-u16): -16 + sum_p w_p beta_p, w = (1, 2, 4, 8, 17), five planes (narrow_u.planes(16),
// offset 16), U in {-16..-1, 1..16}, so W' = W + c + U stays s8 at W in [-110, 110]
__host__ __device__ inline int u_beta16(uint32_t v) { return -16 + (int)(v & 15u) + ((v & 16u) ? 17 : 0); }
__host__ __device__ inline int n_dig(int k) { return k <= 4096 ? 2 : 3; }

// egemm.cu's quant_blk<true> (X and S as there), with P_A's rows S's digits under the carry coins z. DIG 1: that; the
// cost-split stand-ins (wrong P_A): DIG 0 neither z nor digits (P_A row 0 = S's low byte), DIG 2 the digits over a z
// that is zeroed, not hashed. z's digest rides on lane np of the last round's patch hashes when that round leaves a lane
// idle (k / 128 not a multiple of 32: down's k = 14,336), else it is lane 0's own hash first
// H1 (co32c-u16): the per-token amax `amax_row[r]` (crow_s) gives the scale s = amax_row[r] / tq (tq = 126),
// and X divides by it, X = floor(clamp(x / s, lo, 126)) + b, as steps.act_quant does (s = amax / 126, v = (x / s).clamp,
// two roundings, not one reciprocal), clamped on [`lo`, 126] (lo = -126 for H1, -127 for Dither-C's one-scale x * inv_s
// form). Both the per-row s = amax_row / tq and the per-entry x / s divide here, so ptxas shares one generic div.rn slow
// path (the pinned body), not the specialized copy it builds for a function whose only division it can fold.
template <int DIG, bool H1 = false>
DEV void cquant_blk(int blk, uint32_t* dgs, int* sps, uint32_t* zss, const uint16_t* __restrict__ x, int8_t* __restrict__ X,
                    int8_t* __restrict__ PA, int rows, int k, float inv_s, const float* __restrict__ amax_row, float tq, int lo, const Sig& sg) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, grp = blk * 8 + warp;
  if (grp * 4 >= rows) return;
  const int kb = k / 128, m = n_dig(k);
  uint32_t* dg = dgs + 512 * warp;
  int* sp = sps + 128 * warp;
  uint32_t* zs = zss + 16 * warp;   // z's digest, digest byte order: row rr's 128 bits are words 4 rr .. 4 rr + 3
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dg);
  const bool z_alone = kb % 32 == 0;
  if (DIG == 1 && z_alone && lane == 0) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)Z_TAG << 56 | (uint64_t)grp, st);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      zs[2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      zs[2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
  }
  if (DIG == 2 && lane < 16) zs[lane] = 0;
#pragma unroll 1
  for (int p0 = 0; p0 < kb; p0 += 32) {
    const int np = min(32, kb - p0), steps = np / 2;
    const bool zl = DIG == 1 && !z_alone && p0 + 32 >= kb && lane == np;
    if (p0) __syncwarp();
    if (lane < np || zl) {
      uint64_t st[8];
      sha_block(sg.w, zl ? (uint64_t)Z_TAG << 56 | (uint64_t)grp : (uint64_t)DITHER_TAG << 56 | ((uint64_t)grp * kb + p0 + lane), st);
      uint32_t* o = zl ? zs : dg + 16 * lane;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        o[2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
        o[2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
      }
    }
    __syncwarp();
#pragma unroll 1
    for (int rr = 0; rr < 4; rr++) {
      const size_t r = (size_t)grp * 4 + rr;
      const float sc = H1 ? amax_row[r] / tq : inv_s;   // H1: the per-token scale s = amax / 126 to divide by; else inv_s to multiply by
      const uint4* src = reinterpret_cast<const uint4*>(x + r * k) + 16 * p0;
      uint2* dst = reinterpret_cast<uint2*>(X + r * k) + 16 * p0;
      int v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 4
      for (int it = 0; it < steps; it++) {
        const int ch = it * 32 + lane;
        const uint4 w = __ldg(src + ch);
        const uint32_t bits = bytes[64 * (ch / 16) + 16 * rr + ch % 16];
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        int qv[8];
#pragma unroll
        for (int h = 0; h < 4; h++) {
          const float xa = __uint_as_float(ww[h] << 16), xb = __uint_as_float(ww[h] & 0xFFFF0000u);
          float f0, f1;
          if constexpr (H1) { f0 = xa / sc; f1 = xb / sc; } else { f0 = xa * sc; f1 = xb * sc; }
          qv[2 * h] = min(max(__float2int_rd(f0), lo), 126) + (int)((bits >> (2 * h)) & 1u);
          qv[2 * h + 1] = min(max(__float2int_rd(f1), lo), 126) + (int)((bits >> (2 * h + 1)) & 1u);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] += qv[i];
        const uint32_t lo = __byte_perm(__byte_perm(qv[0], qv[1], 0x0040), __byte_perm(qv[2], qv[3], 0x0040), 0x5410);
        const uint32_t hi = __byte_perm(__byte_perm(qv[4], qv[5], 0x0040), __byte_perm(qv[6], qv[7], 0x0040), 0x5410);
        dst[ch] = make_uint2(lo, hi);
      }
      const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int send = b4 ? v[i] : v[i + 4], keep = b4 ? v[i + 4] : v[i];
        v[i] = keep + __shfl_xor_sync(~0u, send, 16);
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const int send = b3 ? v[i] : v[i + 2], keep = b3 ? v[i + 2] : v[i];
        v[i] = keep + __shfl_xor_sync(~0u, send, 8);
      }
      {
        const int send = b2 ? v[0] : v[1], keep = b2 ? v[1] : v[0];
        v[0] = keep + __shfl_xor_sync(~0u, send, 4);
      }
      const int c = 8 * (lane & 3) + (lane >> 2);
      int S = v[0];
      if (kb > 32) {
        if (p0) S += sp[32 * rr + c];
        if (p0 + 32 < kb) { sp[32 * rr + c] = S; continue; }
      }
      if (DIG == 0) {
        int8_t* pr = PA + r * 128;
        pr[c] = (int8_t)S; pr[32 + c] = 0; pr[64 + c] = 0; pr[96 + c] = 0;
        continue;
      }
      // the digits (checked_strip.digits): the last carry is forced to 0 where floor(R / 128) = 127
      const int q0 = S >> 7;
      int z0 = (int)((zs[4 * rr] >> c) & 1u), a0, a1, a2 = 0;
      if (m == 2) {
        if (q0 == 127) z0 = 0;
        a0 = (S & 127) - 128 * z0;
        a1 = q0 + z0;
      } else {
        a0 = (S & 127) - 128 * z0;
        const int r1 = q0 + z0, q1 = r1 >> 7;
        int z1 = (int)((zs[4 * rr + 1] >> c) & 1u);
        if (q1 == 127) z1 = 0;
        a1 = (r1 & 127) - 128 * z1;
        a2 = q1 + z1;
      }
      int8_t* pr = PA + r * 128;
      pr[c] = (int8_t)a0; pr[32 + c] = (int8_t)a1; pr[64 + c] = (int8_t)a2; pr[96 + c] = 0;
    }
  }
}
#define CQUANT(NAME, DIG)                                                                                                  \
  __global__ void __launch_bounds__(256) NAME(const uint16_t* __restrict__ x, int8_t* __restrict__ X, int8_t* __restrict__ PA, \
                                              int rows, int k, float inv_s, const __grid_constant__ Sig sg) {               \
    __shared__ uint32_t dg[8 * 512];                                                                                         \
    __shared__ int sp[8 * 128];                                                                                              \
    __shared__ uint32_t zs[8 * 16];                                                                                          \
    cquant_blk<DIG>(blockIdx.x, dg, sp, zs, x, X, PA, rows, k, inv_s, nullptr, 126.f, -127, sg);                             \
  }
CQUANT(cquant, 1)
CQUANT(cquant_s, 0)
CQUANT(cquant_d, 2)

// H1's per-token amax: amax_row(row) = max(|x_row|, 1e-30) (steps.act_quant at act_top 127, one bit, t = 126). crow_s only
// reduces (no division), so it has no div.rn slow path to pin; cquant_h1 forms s = amax_row / 126 and divides x by s there,
// among its per-entry divides, so every division shares the one pinned generic slow path.
__global__ void __launch_bounds__(256) crow_s(const uint16_t* __restrict__ x, float* __restrict__ amax_row, int rows, int k) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (w >= rows) return;
  const uint16_t* xr = x + (size_t)w * k;
  float a = 0.f;
  for (int i = lane; i < k; i += 32) a = fmaxf(a, fabsf(__uint_as_float((uint32_t)xr[i] << 16)));
#pragma unroll
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(~0u, a, o));
  if (lane == 0) amax_row[w] = fmaxf(a, 1e-30f);
}
// H1's dithered activation quantizer: cquant's digits (DIG 1) dividing x by s = amax_row / 126 and clamping on [-126, 126]
__global__ void __launch_bounds__(256) cquant_h1(const uint16_t* __restrict__ x, int8_t* __restrict__ X, int8_t* __restrict__ PA,
                                                 int rows, int k, const float* __restrict__ amax_row, const __grid_constant__ Sig sg) {
  __shared__ uint32_t dg[8 * 512];
  __shared__ int sp[8 * 128];
  __shared__ uint32_t zs[8 * 16];
  cquant_blk<1, true>(blockIdx.x, dg, sp, zs, x, X, PA, rows, k, 0.f, amax_row, 126.f, -126, sg);
}

// W' = W + c + U and P_B's rows [-U | U | U | 0]: a block 8 rows, its threads loading W's first 16-byte chunks while
// threads 0 .. 2 k / 128 - 1 hash c's patches (two row groups of 4) and the next 4 hash beta's blocks (rows 2j, 2j + 1).
// cs: k bytes of c's digests; us: 64 words of U. H1: the narrow U (u_beta16) over W in [-110, 110].
template <bool H1 = false>
DEV void cwedit_blk(int blk, uint8_t* cs, uint32_t* us, const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp,
                    uint8_t* __restrict__ PB, int k, const Sig& sg) {
  const int r0 = blk * 8, t = threadIdx.x, cpr = k / 16, total = 8 * cpr, kb = k / 128;
  constexpr int MAXC = 8;
  uint4 w[MAXC];
  const uint4* src = reinterpret_cast<const uint4*>(W + (size_t)r0 * k);
#pragma unroll
  for (int i = 0; i < MAXC; i++)
    if (t + 256 * i < total) w[i] = __ldg(src + t + 256 * i);
  if (t < 2 * kb) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)C_TAG << 56 | (uint64_t)((r0 / 4) * kb + t), st);
    uint32_t* d = reinterpret_cast<uint32_t*>(cs + 64 * t);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      d[2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      d[2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
  } else if (t < 2 * kb + 4) {
    const int j = t - 2 * kb;
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)BETA_TAG << 56 | (uint64_t)(r0 / 2 + j), st);
#pragma unroll
    for (int rr = 0; rr < 2; rr++)
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
          const int e = 32 * rr + 4 * q + b;
          const uint32_t db = (uint32_t)(st[e / 8] >> (56 - 8 * (e % 8))) & 0xFFu;
          v |= ((uint32_t)(H1 ? u_beta16(db) : u_beta(db)) & 0xFFu) << (8 * b);
        }
        us[8 * (2 * j + rr) + q] = v;
      }
  }
  __syncthreads();
  {
    const int rr = t / 32, wq = t % 32, u = us[8 * rr + wq % 8];
    const uint32_t nu = add_noise<false>(~(uint32_t)u, 0x01010101u);
    reinterpret_cast<uint32_t*>(PB + (size_t)(r0 + rr) * 128)[wq] = wq / 8 == 0 ? nu : wq / 8 == 3 ? 0u : (uint32_t)u;
  }
  uint4* dst = reinterpret_cast<uint4*>(Wp + (size_t)r0 * k);
#pragma unroll 1
  for (int c0 = 0; c0 < total; c0 += 256 * MAXC) {
    if (c0) {
#pragma unroll
      for (int i = 0; i < MAXC; i++)
        if (c0 + t + 256 * i < total) w[i] = __ldg(src + c0 + t + 256 * i);
    }
#pragma unroll
    for (int i = 0; i < MAXC; i++) {
      const int c = c0 + t + 256 * i;
      if (c < total) {
        const int rr = c / cpr, col = 16 * (c % cpr);
        const uint8_t* cb = cs + 64 * ((rr / 4) * kb + col / 128) + (128 * (rr % 4) + col % 128) / 8;
        const uint32_t b16 = (uint32_t)cb[0] | (uint32_t)cb[1] << 8;
        const uint32_t* u = us + 8 * rr + (col % 32) / 4;
        const uint4 x = w[i];
        dst[c] = make_uint4(add_noise<false>(add_noise<false>(x.x, bits01(b16 & 0xF)), u[0]),
                            add_noise<false>(add_noise<false>(x.y, bits01((b16 >> 4) & 0xF)), u[1]),
                            add_noise<false>(add_noise<false>(x.z, bits01((b16 >> 8) & 0xF)), u[2]),
                            add_noise<false>(add_noise<false>(x.w, bits01(b16 >> 12)), u[3]));
      }
    }
  }
}
constexpr int CW_MAXK = 16128;
__global__ void __launch_bounds__(256) cwedit(const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp, uint8_t* __restrict__ PB, int k,
                                              const __grid_constant__ Sig sg) {
  __shared__ __align__(16) uint8_t cs[CW_MAXK];
  __shared__ uint32_t us[64];
  cwedit_blk<false>(blockIdx.x, cs, us, W, Wp, PB, k, sg);
}
// H1's weight edit: W' = W + c + U with the narrow U (u_beta16); W in [-110, 110], so W' stays s8
__global__ void __launch_bounds__(256) cwedit_h1(const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp, uint8_t* __restrict__ PB, int k,
                                                 const __grid_constant__ Sig sg) {
  __shared__ __align__(16) uint8_t cs[CW_MAXK];
  __shared__ uint32_t us[64];
  cwedit_blk<true>(blockIdx.x, cs, us, W, Wp, PB, k, sg);
}
// W^ = W' - U, the served weights, for the gate's cuBLASLt reference (HALF: cohalf's P_B, -U at byte k % 64)
template <bool HALF> __global__ void phat(const int8_t* __restrict__ Wp, const int8_t* __restrict__ PB, int8_t* __restrict__ Wh, size_t n, int k) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n * k; i += (size_t)gridDim.x * blockDim.x)
    Wh[i] = HALF ? (int8_t)(Wp[i] + PB[(i / k) * 128 + (i % k) % 64]) : (int8_t)(Wp[i] - PB[(i / k) * 128 + 32 + (i % k) % 32]);
}

// ---------------------------------------------------------------- cohalf: QK^T at k = 128, r = k / 2 = 64, Q and K 7-bit
constexpr uint32_t HBETA_TAG = 11;
// the served side (K): X = clamp(floor(x / s), -64, 62) + b (b as cquant's, tag 7, block = row / 4 at k = 128) and
// P_A = [Y_lo | Y_hi | 0 | 0], Y(r) = X(r) + X(r + 64) in [-128, 126]. A block HQ_G row groups of 4: thread g < HQ_G
// hashes group g's digest, then a warp 4 rows at a time, lane l columns 4 l .. 4 l + 3
constexpr int HQ_G = 16;
inline int hq_blocks(int rows) { return (rows + 4 * HQ_G - 1) / (4 * HQ_G); }
__global__ void __launch_bounds__(256) hquant(const uint16_t* __restrict__ x, int8_t* __restrict__ X, int8_t* __restrict__ PA,
                                              int rows, float inv_s, const __grid_constant__ Sig sg) {
  __shared__ uint32_t dgs[HQ_G * 16];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g0 = blockIdx.x * HQ_G;
  if (threadIdx.x < HQ_G && (g0 + (int)threadIdx.x) * 4 < rows) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)DITHER_TAG << 56 | (uint64_t)(g0 + threadIdx.x), st);
    uint32_t* dg = dgs + 16 * threadIdx.x;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      dg[2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      dg[2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int gi = warp; gi < HQ_G && (g0 + gi) * 4 < rows; gi += 8) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dgs + 16 * gi);
#pragma unroll 1
    for (int rr = 0; rr < 4; rr++) {
      const size_t r = (size_t)(g0 + gi) * 4 + rr;
      const uint2 w = __ldg(reinterpret_cast<const uint2*>(x + r * 128) + lane);
      const uint32_t bits = (bytes[16 * rr + lane / 2] >> (4 * (lane & 1))) & 0xFu;
      const uint32_t ww[2] = {w.x, w.y};
      int qv[4];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const float f0 = __uint_as_float(ww[h] << 16) * inv_s, f1 = __uint_as_float(ww[h] & 0xFFFF0000u) * inv_s;
        qv[2 * h] = min(max(__float2int_rd(f0), -64), 62) + (int)((bits >> (2 * h)) & 1u);
        qv[2 * h + 1] = min(max(__float2int_rd(f1), -64), 62) + (int)((bits >> (2 * h + 1)) & 1u);
      }
      const uint32_t xw = __byte_perm(__byte_perm(qv[0], qv[1], 0x0040), __byte_perm(qv[2], qv[3], 0x0040), 0x5410);
      reinterpret_cast<uint32_t*>(X + r * 128)[lane] = xw;
      const uint32_t hi = __shfl_down_sync(~0u, xw, 16);
      reinterpret_cast<uint32_t*>(PA + r * 128)[lane] = lane < 16 ? add_noise<false>(xw, hi) : 0u;
    }
  }
}
// H1's per (head, token) activation extremes for cohalf's QK^T: amax = max(x), namax = max(-x) over the 128 head-dim
// (steps.floor_quant's x.amax and (-x).amax at top 111), packed as (amax, namax). A separate reduction (no division), so
// hquant_h1's divides share the one pinned generic div.rn slow path, as crow_s does for cquant_h1 (an inline reduction
// interleaves warp barriers that make ptxas emit an unpinned div slow-path body).
__global__ void __launch_bounds__(256) hrow_s(const uint16_t* __restrict__ x, float2* __restrict__ an, int rows) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (w >= rows) return;
  const uint2 v = __ldg(reinterpret_cast<const uint2*>(x + (size_t)w * 128) + lane);
  const uint32_t vv[2] = {v.x, v.y};
  float pa = __uint_as_float(0xff800000u), na = pa;   // -inf
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const float a = __uint_as_float(vv[h] << 16), b = __uint_as_float(vv[h] & 0xFFFF0000u);
    pa = fmaxf(fmaxf(pa, a), b);
    na = fmaxf(fmaxf(na, -a), -b);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { pa = fmaxf(pa, __shfl_xor_sync(~0u, pa, o)); na = fmaxf(na, __shfl_xor_sync(~0u, na, o)); }
  if (lane == 0) an[w] = make_float2(pa, na);
}
// H1's served side (K): per (head, token) asymmetric floor at top 111 (steps.floor_quant at top 111, one bit): with (amax,
// namax) = hrow_s(x), s = max(amax / 111, namax / 112) clamped to 1e-30 then * (1 + 2^-20), X = floor(x / s) + b in
// [-112, 111] (the margin keeps floor(x / s) in [-112, 110], no clamp). Y(r) = X(r) + X(r + 64) in [-224, 222] leaves s8,
// so the digit split Y = Y0 + 128 Y1 (Y0 = ((Y + 64) & 127) - 64 in [-64, 63], Y1 = (Y - Y0) >> 7 in [-2, 2]) and
// P_A = [Y0_lo | Y0_hi | Y1_lo | Y1_hi]. The scale read from memory keeps every divide (amax / 111, namax / 112, x / s)
// on the one pinned generic div.rn slow path.
__global__ void __launch_bounds__(256) hquant_h1(const uint16_t* __restrict__ x, int8_t* __restrict__ X, int8_t* __restrict__ PA,
                                                 int rows, const float2* __restrict__ an, const __grid_constant__ Sig sg) {
  __shared__ uint32_t dgs[HQ_G * 16];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g0 = blockIdx.x * HQ_G;
  if (threadIdx.x < HQ_G && (g0 + (int)threadIdx.x) * 4 < rows) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)DITHER_TAG << 56 | (uint64_t)(g0 + threadIdx.x), st);
    uint32_t* dg = dgs + 16 * threadIdx.x;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      dg[2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      dg[2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int gi = warp; gi < HQ_G && (g0 + gi) * 4 < rows; gi += 8) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dgs + 16 * gi);
#pragma unroll 1
    for (int rr = 0; rr < 4; rr++) {
      const size_t r = (size_t)(g0 + gi) * 4 + rr;
      const uint2 w = __ldg(reinterpret_cast<const uint2*>(x + r * 128) + lane);
      const uint32_t ww[2] = {w.x, w.y};
      float xf[4];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        xf[2 * h] = __uint_as_float(ww[h] << 16);
        xf[2 * h + 1] = __uint_as_float(ww[h] & 0xFFFF0000u);
      }
      const float2 pn = __ldg(an + r);
      const float s = fmaxf(fmaxf(pn.x / 111.f, pn.y / 112.f), 1e-30f) * (1.f + 0x1p-20f);
      const uint32_t bits = (bytes[16 * rr + lane / 2] >> (4 * (lane & 1))) & 0xFu;
      int qv[4];
#pragma unroll
      for (int j = 0; j < 4; j++) qv[j] = __float2int_rd(xf[j] / s) + (int)((bits >> j) & 1u);
      const uint32_t xw = __byte_perm(__byte_perm(qv[0], qv[1], 0x0040), __byte_perm(qv[2], qv[3], 0x0040), 0x5410);
      reinterpret_cast<uint32_t*>(X + r * 128)[lane] = xw;
      const uint32_t hi = __shfl_down_sync(~0u, xw, 16);
      uint32_t y0w = 0, y1w = 0;
#pragma unroll
      for (int b = 0; b < 4; b++) {
        const int Y = (int)(int8_t)(xw >> (8 * b)) + (int)(int8_t)(hi >> (8 * b));
        const int Y0 = ((Y + 64) & 127) - 64, Y1 = (Y - Y0) >> 7;
        y0w |= ((uint32_t)(uint8_t)(int8_t)Y0) << (8 * b);
        y1w |= ((uint32_t)(uint8_t)(int8_t)Y1) << (8 * b);
      }
      // P_A row = [Y0 (cols 0..63) | Y1 (cols 0..63)]: lanes 0..15 hold Y0/Y1, a uniform store (lane l writes word l, Y1
      // pulled from lane l - 16) in place of a lane < 16 branch. The shuffle runs on every lane (a mask that any lane skips
      // deadlocks); lanes < 16 take their Y0 and drop the shuffled word
      const uint32_t y1from = __shfl_sync(~0u, y1w, lane - 16);
      reinterpret_cast<uint32_t*>(PA + r * 128)[lane] = lane < 16 ? y0w : y1from;
    }
  }
}
// the peeled side (Q in [-64, 62]): Q' = Q + c + U(i, k mod 64) (c as cwedit's, tag 8; U(i, r) from digest byte r of
// row i's tag-11 block, r < 64, as cwedit's beta) and P_B = [-U_lo | -U_hi | 0 | 0]. A block HW_G row groups of 4:
// thread t < HW_G hashes group t's c, thread HW_G + j row j's U (5 HW_G digests on 5 full warps at HW_G = 32), then a
// warp 4 rows at a time. WH: also W^ = Q + c (phat<true>'s bytes) to Wh
constexpr int HW_G = 32;
inline int hw_blocks(int rows) { return (rows + 4 * HW_G - 1) / (4 * HW_G); }
// H1 (co32c-u16): the narrow U (u_beta16) over Q in [-112, 110] (floor top 111), so Q' = Q + c + U stays s8 at Q + c in
// [-112, 111]; P_B = [-U_lo | -U_hi | -U_lo | -U_hi] (the repeat for the digit split's Y1 atoms, htail_h1)
template <bool WH = false, bool H1 = false>
__global__ void __launch_bounds__(256) hwedit(const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp, uint8_t* __restrict__ PB, int rows,
                                              const __grid_constant__ Sig sg, uint8_t* __restrict__ Wh = nullptr) {
  __shared__ uint32_t cds[HW_G * 16];
  __shared__ uint32_t uds[HW_G * 64];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g0 = blockIdx.x * HW_G, t = threadIdx.x;
  if (t < 5 * HW_G && (g0 + (t < HW_G ? t : (t - HW_G) / 4)) * 4 < rows) {
    uint64_t st[8];
    sha_block(sg.w, t < HW_G ? ((uint64_t)C_TAG << 56 | (uint64_t)(g0 + t)) : ((uint64_t)HBETA_TAG << 56 | (uint64_t)(4 * g0 + t - HW_G)), st);
    if (t < HW_G) {
      uint32_t* cd = cds + 16 * t;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        cd[2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
        cd[2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
      }
    } else {
      uint32_t* ud = uds + 16 * (t - HW_G);
#pragma unroll
      for (int q = 0; q < 16; q++) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
          const int e = 4 * q + b;
          const uint32_t db = (uint32_t)(st[e / 8] >> (56 - 8 * (e % 8))) & 0xFFu;
          v |= ((uint32_t)(H1 ? u_beta16(db) : u_beta(db)) & 0xFFu) << (8 * b);
        }
        ud[q] = v;
      }
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int gi = warp; gi < HW_G && (g0 + gi) * 4 < rows; gi += 8) {
    const uint8_t* cb = reinterpret_cast<const uint8_t*>(cds + 16 * gi);
    const uint32_t* ud = uds + 64 * gi;
#pragma unroll 1
    for (int rr = 0; rr < 4; rr++) {
      const size_t r = (size_t)(g0 + gi) * 4 + rr;
      const uint32_t w = __ldg(reinterpret_cast<const uint32_t*>(W + r * 128) + lane);
      const uint32_t nib = (cb[16 * rr + lane / 2] >> (4 * (lane & 1))) & 0xFu, u = ud[16 * rr + (lane & 15)];
      const uint32_t wc = add_noise<false>(w, bits01(nib));
      reinterpret_cast<uint32_t*>(Wp + r * 128)[lane] = add_noise<false>(wc, u);
      if constexpr (WH) reinterpret_cast<uint32_t*>(Wh + r * 128)[lane] = wc;
      reinterpret_cast<uint32_t*>(PB + r * 128)[lane] = (H1 || lane < 16) ? add_noise<false>(~u, 0x01010101u) : 0u;
    }
  }
}
// pv's served epilogue for the gate: F = float(2 O + 127 S_row) (s_row s_col), c_body's SERVE 2 op for op
__global__ void vref(const int* __restrict__ O, const uint8_t* __restrict__ eb, float* __restrict__ F, int M, int N) {
  const int* S = reinterpret_cast<const int*>(eb);
  const float* sr = reinterpret_cast<const float*>(S + M);
  const float* sc = sr + M;
  const int r = blockIdx.y;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < N; c += gridDim.x * blockDim.x)
    F[(size_t)r * N + c] = __int2float_rn(2 * O[(size_t)r * N + c] + 127 * S[r]) * (sr[r] * sc[c]);
}
// eb: S in [-32768, 32767], the scales in [2^-10, 2^-9)
__global__ void ebfill(uint8_t* eb, int M, int N, uint32_t seed) {
  int* S = reinterpret_cast<int*>(eb);
  float* s = reinterpret_cast<float*>(S + M);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 2 * M + N; i += gridDim.x * blockDim.x) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)i));
    if (i < M) S[i] = (int)(h & 0xFFFFu) - 32768;
    else s[i - M] = __uint_as_float(0x3A800000u | (h & 0x7FFFFFu));
  }
}
// W in [lo, hi] (co32c: [-64, 62]; H1: [-110, 110]); the gate's rows 0 and 1 at hi and lo
__global__ void wfill(int8_t* W, size_t count, uint32_t seed, size_t k, int ends, int lo, int hi) {
  const uint32_t span = (uint32_t)(hi - lo + 1);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)i) ^ mix32((uint32_t)(i >> 32) + 0x55u));
    int v = (int)(h % span) + lo;
    if (ends && i < k) v = hi;
    else if (ends && i < 2 * k) v = lo;
    W[i] = (int8_t)v;
  }
}

// ---------------------------------------------------------------- the GEMM's tail
// tile position J's words into chains 4 (J % 2) + e: duo's fold (RK DUO), or for an fx kind duo's fold of egemm.cu's
// fx_chain<RK, J, e> of the word (FX1: one IMAD by key_mul(1, (4 J + e) 32), so f = 4)
template <class L, int RK, int J> DEV void cfold(St (&ch)[8], const int (&w)[4]) {
  if constexpr (is_fx(RK)) {
    constexpr int H = 4 * (J % 2);
    duo<DUO, 0, J>(ch[H], fx_chain<RK, J, 0>((uint32_t)w[0]));
    duo<DUO, 0, J>(ch[H + 1], fx_chain<RK, J, 1>((uint32_t)w[1]));
    duo<DUO, 0, J>(ch[H + 2], fx_chain<RK, J, 2>((uint32_t)w[2]));
    duo<DUO, 0, J>(ch[H + 3], fx_chain<RK, J, 3>((uint32_t)w[3]));
  } else {
    static_assert(RK == DUO, "co32c's tail folds with duo or an fx kind");
    L::template absorb<DUO, J / L::NI, J % L::NI>(ch, w);
  }
}
// step D: D_d = A_d U^T tile by tile (atom D of the tail stage), each absorbed as made, and (SERVE) O -= 128^d D_d
template <class L, int D, bool SERVE, int RK = DUO, int J = 0>
DEV void cdstep(int (&acc)[L::MI][L::NI][4], St (&ch)[8], const uint32_t (&fa)[L::MI][4], const uint32_t (&fb)[L::NI][2]) {
  if constexpr (J < L::MI * L::NI) {
    constexpr int mi = J / L::NI, ni = J % L::NI;
    int d[4];
    imma0(d, fa[mi], fb[ni]);
    cfold<L, RK, J>(ch, d);
    if constexpr (SERVE) {
#pragma unroll
      for (int e = 0; e < 4; e++) acc[mi][ni][e] = (int)((uint32_t)acc[mi][ni][e] - ((uint32_t)d[e] << (7 * D)));
    }
    cdstep<L, D, SERVE, RK, J + 1>(acc, ch, fa, fb);
  }
}
template <class L, int RK = DUO, int J = 0> DEV void cabs_all(St (&ch)[8], const int (&w)[L::MI][L::NI][4]) {
  if constexpr (J < L::MI * L::NI) {
    cfold<L, RK, J>(ch, w[J / L::NI][J % L::NI]);
    cabs_all<L, RK, J + 1>(ch, w);
  }
}
// the tail stage P_A = [A_0 | A_1 | A_2 | 0], P_B = [-U | U | U | 0]: O' = T_last + A_0 (-U)^T (atom 0, accumulated onto
// T_last), absorbed; then D_1 (atom 1) and with THREE D_2 (atom 2)
template <class L, bool THREE, bool SERVE, int RK = DUO> DEV void ctail(L& ml, int (&acc)[L::MI][L::NI][4], St (&ch)[8]) {
  constexpr int MI = L::MI, NI = L::NI;
  ml.wait_full();
  const uint32_t tb = ml.stage(ml.s);
  uint32_t fa[MI][4], fb[NI][2];
  ml.load_b(fb, tb, 0);
#pragma unroll
  for (int mi = 0; mi < MI; mi++) {
    ml.load_a(fa[mi], tb, 0, mi);
#pragma unroll
    for (int ni = 0; ni < NI; ni++) imma(acc[mi][ni], fa[mi], fb[ni]);
  }
  cabs_all<L, RK>(ch, acc);
  ml.load_b(fb, tb, 1);
#pragma unroll
  for (int mi = 0; mi < MI; mi++) ml.load_a(fa[mi], tb, 1, mi);
  cdstep<L, 1, SERVE, RK>(acc, ch, fa, fb);
  if constexpr (THREE) {
    ml.load_b(fb, tb, 2);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) ml.load_a(fa[mi], tb, 2, mi);
    cdstep<L, 2, SERVE, RK>(acc, ch, fa, fb);
  }
  ml.release(ml.empty(ml.s));
  ml.advance();
}
// cohalf's tail stage P_A = [Y_lo | Y_hi | 0 | 0], P_B = [-U_lo | -U_hi | 0 | 0]: O = T_last - U Y (atoms 0 and 1
// accumulated onto T_last), absorbed
template <class L> DEV void htail(L& ml, int (&acc)[L::MI][L::NI][4], St (&ch)[8]) {
  constexpr int MI = L::MI, NI = L::NI;
  ml.wait_full();
  const uint32_t tb = ml.stage(ml.s);
  uint32_t fa[MI][4], fb[NI][2];
#pragma unroll
  for (int kk = 0; kk < 2; kk++) {
    ml.load_b(fb, tb, kk);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      ml.load_a(fa[mi], tb, kk, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) imma(acc[mi][ni], fa[mi], fb[ni]);
    }
  }
  cabs_all<L>(ch, acc);
  ml.release(ml.empty(ml.s));
  ml.advance();
}
// cohalf H1's tail stage P_A = [Y0_lo | Y0_hi | Y1_lo | Y1_hi], P_B = [-U_lo | -U_hi | -U_lo | -U_hi]: Y = X_lo + X_hi in
// [-224, 222] = Y0 + 128 Y1, so O = T_last + Y (-U) = T_last + Y0 (-U) + 128 (Y1 (-U)). Atoms 0 and 1 (Y0) accumulate onto
// T_last; atoms 2 and 3 (Y1) accumulate into a fresh d1, folded in at base 128 (as cdstep's << 7, but added not subtracted)
template <class L> DEV void htail_h1(L& ml, int (&acc)[L::MI][L::NI][4], St (&ch)[8]) {
  constexpr int MI = L::MI, NI = L::NI;
  ml.wait_full();
  const uint32_t tb = ml.stage(ml.s);
  uint32_t fa[MI][4], fb[NI][2];
#pragma unroll
  for (int kk = 0; kk < 2; kk++) {
    ml.load_b(fb, tb, kk);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      ml.load_a(fa[mi], tb, kk, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) imma(acc[mi][ni], fa[mi], fb[ni]);
    }
  }
  int d1[MI][NI][4];
#pragma unroll
  for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int ni = 0; ni < NI; ni++)
#pragma unroll
      for (int e = 0; e < 4; e++) d1[mi][ni][e] = 0;
#pragma unroll
  for (int kk = 2; kk < 4; kk++) {
    ml.load_b(fb, tb, kk);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      ml.load_a(fa[mi], tb, kk, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) imma(d1[mi][ni], fa[mi], fb[ni]);
    }
  }
#pragma unroll
  for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int ni = 0; ni < NI; ni++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[mi][ni][e] += d1[mi][ni][e] * 128;
  cabs_all<L>(ch, acc);
  ml.release(ml.empty(ml.s));
  ml.advance();
}

// TAIL 0: co32c's (ctail), 1: cohalf's (htail), 2: cohalf H1's (htail_h1). SERVE 0: O' to U; 1: O; 2: P.V's served value as float, f = float(2 O + 127 S_row) (s_row s_col) with eb = [S (M ints) |
// s_row (M floats) | s_col (N floats)] (pv's epilogue on a full key block, S per channel: the epilogue's cost a cell).
// RK: the fold, DUO or (co32c) an fx kind on every absorbed word, the mainloop's (EL's) and the tail's
template <int LAG, int TIE, bool SLEEP, bool THREE, int SERVE, int TAIL = 0, int RK = DUO> DEV void c_body(const Args& a) {
  static_assert(RK == DUO || (TAIL == 0 && is_fx(RK)), "an fx fold is co32c's");
  using L = EL<RK, LAG, TIE, SLEEP>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps, a.M, a.N, a.K, a.group, true, 1);
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int tiles_m = a.M / 128, tiles_n = a.N / 128;
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  const uint32_t zero = (uint32_t)a.KR;
  uint32_t kr[L::NKR];
  kr[0] = 0;
  typename L::Place pl{0, 0};
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    ml.place_step(pl, t == (int)blockIdx.x ? t : (int)gridDim.x);
    L::place_tile(pl, tiles_m, tm, tn);
    const uint32_t tile_id = tm * tiles_n + tn;
    St ch[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      ch[e].y = mix32(a.seed ^ mix32((tile_id * L::NCT + ctid) * 8 + e));
      ch[e].z = mix32(ch[e].y + 0x9E3779B9u);
    }
    int acc[MI][NI][4];
    ml.tile_il(a.K, acc, ch, kr, zero);
    if constexpr (TAIL == 1) htail<L>(ml, acc, ch);
    else if constexpr (TAIL == 2) htail_h1<L>(ml, acc, ch);
    else ctail<L, THREE, SERVE != 0, RK>(ml, acc, ch);
    const int row0 = tm * 128 + wm * 32 + g, col0 = tn * 128 + wn * 32 + 2 * q;
    if constexpr (SERVE == 2) {
      const int* S = reinterpret_cast<const int*>(a.eb);
      const float* sr = reinterpret_cast<const float*>(S + a.M);
      const float* sc = sr + a.M;
      int sv[MI][2];
      float rs[MI][2];
      float2 cs[NI];
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int h = 0; h < 2; h++) { sv[mi][h] = __ldg(S + row0 + 16 * mi + 8 * h); rs[mi][h] = __ldg(sr + row0 + 16 * mi + 8 * h); }
#pragma unroll
      for (int ni = 0; ni < NI; ni++) cs[ni] = __ldg(reinterpret_cast<const float2*>(sc + col0 + 8 * ni));
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          float f[4];
#pragma unroll
          for (int e = 0; e < 4; e++)
            f[e] = __int2float_rn(2 * acc[mi][ni][e] + 127 * sv[mi][e / 2]) * (rs[mi][e / 2] * (e % 2 ? cs[ni].y : cs[ni].x));
          float* p = reinterpret_cast<float*>(a.U) + (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni;
          *reinterpret_cast<float2*>(p) = make_float2(f[0], f[1]);
          *reinterpret_cast<float2*>(p + (size_t)8 * a.N) = make_float2(f[2], f[3]);
        }
    } else {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          int* p = a.U + (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni;
          *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
          *reinterpret_cast<int2*>(p + (size_t)8 * a.N) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
    uint4* r = reinterpret_cast<uint4*>(a.R) + ((size_t)tile_id * L::NCT + ctid) * 2;
    r[0] = make_uint4(finish<DUO>(ch, 0), finish<DUO>(ch, 1), finish<DUO>(ch, 2), finish<DUO>(ch, 3));
    r[1] = make_uint4(finish<DUO>(ch, 4), finish<DUO>(ch, 5), finish<DUO>(ch, 6), finish<DUO>(ch, 7));
  }
}
// NAME (m = 2) and NAME_k3 (m = 3, k > 4096); _o: O' written instead of O
#define CGEMM(NAME, LAG, TIE, SLEEP)                                                                                                \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ Args a) { c_body<LAG, TIE, SLEEP, false, 1>(a); }     \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME##_k3(const __grid_constant__ Args a) { c_body<LAG, TIE, SLEEP, true, 1>(a); } \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME##_o(const __grid_constant__ Args a) { c_body<LAG, TIE, SLEEP, false, 0>(a); }  \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME##_o_k3(const __grid_constant__ Args a) { c_body<LAG, TIE, SLEEP, true, 0>(a); }
CGEMM(c_l7qs, 7, 4, true)
CGEMM(c_l2q, 2, 4, false)
// _v: pv's served epilogue (SERVE 2)
__global__ void __launch_bounds__(MLW::NT, 1) c_l7qs_v(const __grid_constant__ Args a) { c_body<7, 4, true, false, 2>(a); }
__global__ void __launch_bounds__(MLW::NT, 1) c_l7qs_v_k3(const __grid_constant__ Args a) { c_body<7, 4, true, true, 2>(a); }
// c_fx1: c_l7qs with every absorbed word behind fx1's keyed IMAD (the fx1_l7qs chain and keys)
__global__ void __launch_bounds__(MLW::NT, 1) c_fx1(const __grid_constant__ Args a) { c_body<7, 4, true, false, 1, 0, FX1>(a); }
__global__ void __launch_bounds__(MLW::NT, 1) c_fx1_k3(const __grid_constant__ Args a) { c_body<7, 4, true, true, 1, 0, FX1>(a); }
// cohalf (k = 128)
__global__ void __launch_bounds__(MLW::NT, 1) h_l7qs(const __grid_constant__ Args a) { c_body<7, 4, true, false, 1, 1>(a); }
__global__ void __launch_bounds__(MLW::NT, 1) h_l2q(const __grid_constant__ Args a) { c_body<2, 4, false, false, 1, 1>(a); }
// cohalf H1 (co32c-u16): Q/K floored at top 111, the digit-split tail (htail_h1, TAIL 2)
__global__ void __launch_bounds__(MLW::NT, 1) h_l7qs_h1(const __grid_constant__ Args a) { c_body<7, 4, true, false, 1, 2>(a); }
__global__ void __launch_bounds__(MLW::NT, 1) h_l2q_h1(const __grid_constant__ Args a) { c_body<2, 4, false, false, 1, 2>(a); }

// ---------------------------------------------------------------- host
struct CKn { const char* name; Kern k, k3; };
static const CKn CKERN[] = {{"c_l7qs", c_l7qs, c_l7qs_k3}, {"c_l7qs_o", c_l7qs_o, c_l7qs_o_k3}, {"c_l2q", c_l2q, c_l2q_k3},
                          {"c_l2q_o", c_l2q_o, c_l2q_o_k3}, {"c_l7qs_v", c_l7qs_v, c_l7qs_v_k3}, {"c_fx1", c_fx1, c_fx1_k3},
                          {"h_l7qs", h_l7qs, h_l7qs}, {"h_l2q", h_l2q, h_l2q},
                          {"h_l7qs_h1", h_l7qs_h1, h_l7qs_h1}, {"h_l2q_h1", h_l2q_h1, h_l2q_h1}};
static const CKn* ckern(const std::string& n) {
  for (auto& e : CKERN) if (n == e.name) return &e;
  return nullptr;
}
// egemm.cu's EP (its operands, cuBLASLt plans and arms) and co32c's own: W in [-64, 62], X and P_A from cquant, W' and
// P_B from cwedit, W^ for the reference; at k = 128 (half) cohalf's: X and P_A from hquant (at inv_s / 2: 7 bits), W'
// and P_B from hwedit. eb: c_l7qs_v's S and scales
struct CP {
  EP E;
  int8_t *W7, *X, *PA, *Wp, *PB, *Wh;
  int *Y, *Yr;
  uint32_t *R, *R2;
  uint8_t* eb;
  Maps* mC;
  LtG ref;
  uint32_t inst = 0;
  bool half = false;
  bool h1 = false;    // H1's numerics (co32c-u16): per-token scales, +-110 weights, the narrow U
  float inv7 = 0;
  float* amax_row = nullptr;   // H1's per-token activation amax (max|x|, clamped), filled by crow_s; cquant_h1 forms amax/126
  float2* hqa = nullptr;       // cohalf H1's per (head, token) (max(x), max(-x)), filled by hrow_s; hquant_h1 forms amax/111, namax/112
};
using CqK = void (*)(const uint16_t*, int8_t*, int8_t*, int, int, float, Sig);
static void cq(CP& C, const Sig& s, CqK f = cquant) {
  EP& P = C.E;
  if (C.half) {
    if (C.h1) {
      hrow_s<<<(P.m + 7) / 8, 256, 0, P.st>>>(P.x, C.hqa, P.m);
      hquant_h1<<<hq_blocks(P.m), 256, 0, P.st>>>(P.x, C.X, C.PA, P.m, C.hqa, s);
    } else hquant<<<hq_blocks(P.m), 256, 0, P.st>>>(P.x, C.X, C.PA, P.m, C.inv7, s);
    return;
  }
  if (C.h1) {
    crow_s<<<(P.m + 7) / 8, 256, 0, P.st>>>(P.x, C.amax_row, P.m, P.k);
    cquant_h1<<<(P.m + QROWS - 1) / QROWS, 256, 0, P.st>>>(P.x, C.X, C.PA, P.m, P.k, C.amax_row, s);
    return;
  }
  f<<<(P.m + QROWS - 1) / QROWS, 256, 0, P.st>>>(P.x, C.X, C.PA, P.m, P.k, P.inv_s, s);
}
static void cw(CP& C, const Sig& s) {
  EP& P = C.E;
  const uint8_t* W = reinterpret_cast<const uint8_t*>(C.W7);
  uint8_t* Wp = reinterpret_cast<uint8_t*>(C.Wp);
  uint8_t* PB = reinterpret_cast<uint8_t*>(C.PB);
  if (C.half) {
    if (C.h1) hwedit<false, true><<<hw_blocks(P.n), 256, 0, P.st>>>(W, Wp, PB, P.n, s);
    else hwedit<<<hw_blocks(P.n), 256, 0, P.st>>>(W, Wp, PB, P.n, s);
    return;
  }
  if (C.h1) cwedit_h1<<<P.n / 8, 256, 0, P.st>>>(W, Wp, PB, P.k, s);
  else cwedit<<<P.n / 8, 256, 0, P.st>>>(W, Wp, PB, P.k, s);
}
static Kern cpick(const CP& C, const CKn* e) { return n_dig(C.E.k) == 3 ? e->k3 : e->k; }
static void cgemm_run(CP& C, Kern kern, int* Y, uint32_t* R) {
  EP& P = C.E;
  Args a{C.mC, Y, R, P.m, P.n, P.k, 8, P.seed, 0, 1, 0, 0, {0, 0, 0, 0}, C.eb, 0};
  kern<<<P.grid, MLW::NT, CSMEM, P.st>>>(a);
}
static void csetup(CP& C, int m, int n, int k, uint32_t seed, bool gate) {
  EP& P = C.E;
  esetup(P, m, n, k, seed, gate);
  C.half = k == 128;
  const char* num = getenv("CG_NUMERICS");
  C.h1 = num && !strcmp(num, "h1");   // H1's numerics: the linears (k > 128) and cohalf's QK^T (k = 128)
  C.inv7 = P.inv_s * 0.5f;
  CK(cudaMalloc(&C.W7, (size_t)n * k)); CK(cudaMalloc(&C.X, (size_t)m * k)); CK(cudaMalloc(&C.PA, (size_t)m * 128));
  CK(cudaMalloc(&C.Wp, (size_t)n * k)); CK(cudaMalloc(&C.PB, (size_t)n * 128)); CK(cudaMalloc(&C.Wh, (size_t)n * k));
  CK(cudaMalloc(&C.Y, P.ny * 4)); CK(cudaMalloc(&C.Yr, P.ny * 4));
  CK(cudaMalloc(&C.R, P.nr * 4)); CK(cudaMalloc(&C.R2, P.nr * 4));
  CK(cudaMalloc(&C.eb, (size_t)(2 * m + n) * 4));
  if (C.h1 && !C.half) CK(cudaMalloc(&C.amax_row, (size_t)m * 4));
  if (C.h1 && C.half) CK(cudaMalloc(&C.hqa, (size_t)m * sizeof(float2)));
  ebfill<<<256, 256, 0, P.st>>>(C.eb, m, n, mix32(seed ^ 0x7777u));
  // H1: the linears' W in [-110, 110]; cohalf's pre-bit query in [-112, 110] (floor top 111). co32c: [-64, 62]
  const int wlo = C.h1 ? (C.half ? -112 : -110) : -64, whi = C.h1 ? 110 : 62;
  wfill<<<4096, 256, 0, P.st>>>(C.W7, (size_t)n * k, mix32(seed ^ 0x6666u), k, gate, wlo, whi);
  const Sig s0 = sigma(seed, 0xFFFFFFFFu);
  cq(C, s0);
  cw(C, s0);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  C.mC = upload_maps(Maps{tmap(C.X, k, m, 128), tmap(C.Wp, k, n, 128), tmap(C.PA, 128, m, 128), tmap(C.PB, 128, n, 128)});
  C.ref = ltg(I8, m, n, k, k, k, n, C.X, C.Wh, C.Yr, 0, P.st);
  for (auto& e : CKERN)
    for (Kern kern : {e.k, e.k3}) CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
}
static void cfree(CP& C) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)C.W7, (void*)C.X, (void*)C.PA, (void*)C.Wp, (void*)C.PB, (void*)C.Wh, (void*)C.Y, (void*)C.Yr, (void*)C.R,
                  (void*)C.R2, (void*)C.eb, (void*)C.mC, (void*)C.amax_row, (void*)C.hqa})
    CK(cudaFree(p));
  efree(C.E);
}
// an arm: a c_* kernel (on cquant's and cwedit's operands), cquant (cquant_s, cquant_d: its cost-split stand-ins), cwedit,
// cedited:KERN (cquant, cwedit, KERN under one key), or any of egemm.cu's arms (plain, plainq, e_l7qs, editedq:e_l7qs, ...)
// on its own operands
static std::function<void(CP&)> carm(const std::string& a) {
  if (a == "cquant") return [](CP& C) { cq(C, sigma(C.E.seed, C.inst++)); };
  if (a == "cquant_s") return [](CP& C) { cq(C, sigma(C.E.seed, C.inst++), cquant_s); };
  if (a == "cquant_d") return [](CP& C) { cq(C, sigma(C.E.seed, C.inst++), cquant_d); };
  if (a == "cwedit") return [](CP& C) { cw(C, sigma(C.E.seed, C.inst++)); };
  if (a.rfind("cedited:", 0) == 0) {
    const CKn* k = ckern(a.substr(8));
    if (!k) { fprintf(stderr, "no kernel %s\n", a.c_str()); exit(2); }
    return [k](CP& C) {
      const Sig s = sigma(C.E.seed, C.inst++);
      cq(C, s); cw(C, s); cgemm_run(C, cpick(C, k), C.Y, C.R);
    };
  }
  if (const CKn* k = ckern(a)) return [k](CP& C) { cgemm_run(C, cpick(C, k), C.Y, C.R); };
  auto f = earm(a);
  return [f](CP& C) { f(C.E); };
}

static int cgate(const std::string& dir, int m, int n, int k, uint32_t seed) {
  CP C;
  csetup(C, m, n, k, seed, true);
  EP& P = C.E;
  const size_t xb = (size_t)m * k, wb = (size_t)n * k;
  for (auto [p, b] : {std::pair<void*, size_t>{C.X, xb}, {C.PA, (size_t)m * 128}, {C.Wp, wb}, {C.PB, (size_t)n * 128}, {C.Wh, wb},
                      {C.Y, P.ny * 4}, {C.Yr, P.ny * 4}, {C.R, P.nr * 4}, {C.R2, P.nr * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, P.st));
  const Sig s = sigma(seed, 0);
  cq(C, s);
  cw(C, s);
  if (C.half) phat<true><<<4096, 256, 0, P.st>>>(C.Wp, C.PB, C.Wh, (size_t)n, k);
  else phat<false><<<4096, 256, 0, P.st>>>(C.Wp, C.PB, C.Wh, (size_t)n, k);
  ltg_run(C.ref, P.st);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  bool ok = true;
  std::string res;
  auto rec = [&](const std::string& name, unsigned long long y, long long r) {
    res += (res.empty() ? "" : ", ") + std::string("\"") + name + "\": {\"y_bad\": " + std::to_string(y) + ", \"r_bad\": " + std::to_string(r) + "}";
  };
  const int tiles_m = m / 128, tiles_n = n / 128;
  const int tl[3][2] = {{0, 0}, {tiles_m / 2, tiles_n / 2}, {tiles_m - 1, tiles_n - 1}};
  auto dump_tiles = [&](const std::string& name, const int* Y, const uint32_t* R) {
    for (int i = 0; i < 3; i++) {
      const int tm = tl[i][0], tn = tl[i][1];
      const std::string sfx = "_" + std::to_string(tm) + "_" + std::to_string(tn) + ".bin";
      if (R) dump(dir, ("R_" + name + sfx).c_str(), R + (size_t)(tm * tiles_n + tn) * 512 * 8, 512 * 8 * 4, true);
      std::vector<int> yt(128 * 128);
      for (int r = 0; r < 128; r++)
        CK(cudaMemcpy(yt.data() + r * 128, Y + (size_t)(tm * 128 + r) * n + tn * 128, 128 * 4, cudaMemcpyDeviceToHost));
      dump(dir, ("Y_" + name + sfx).c_str(), yt.data(), yt.size() * 4, false);
    }
  };
  // the main kernel (c_l7qs, cohalf's h_l7qs): O against cuBLASLt's X W^T; its receipts are the reference for the
  // other schedule (c_l2q, h_l2q) and the other outputs (c_*_o: O', c_l7qs_v: pv's served epilogue against vref's)
  const std::string km = C.half ? (C.h1 ? "h_l7qs_h1" : "h_l7qs") : "c_l7qs";
  CK(cudaMemsetAsync(C.Y, 0xA5, P.ny * 4, P.st));
  cgemm_run(C, cpick(C, ckern(km)), C.Y, C.R2);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  {
    const unsigned long long y = ecmp(P, C.Y, C.Yr, P.ny);
    rec(km, y, -1);
    ok &= y == 0;
    dump_tiles(km, C.Y, C.R2);
  }
  float* F = nullptr;
  if (!C.half) {
    CK(cudaMalloc(&F, P.ny * 4));
    vref<<<dim3((n + 255) / 256, m), 256, 0, P.st>>>(C.Yr, C.eb, F, m, n);
  }
  const std::vector<std::string> others =
      C.half ? std::vector<std::string>{C.h1 ? "h_l2q_h1" : "h_l2q"} : std::vector<std::string>{"c_l2q", "c_l7qs_o", "c_l2q_o", "c_l7qs_v"};
  for (const std::string& name : others) {
    const bool o = name.find("_o") != std::string::npos, v = name.find("_v") != std::string::npos;
    CK(cudaMemsetAsync(C.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(C.R, 0xA5, P.nr * 4, P.st));
    cgemm_run(C, cpick(C, ckern(name)), C.Y, C.R);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    const unsigned long long y = o ? 0 : ecmp(P, C.Y, v ? (const void*)F : (const void*)C.Yr, P.ny), r = ecmp(P, C.R, C.R2, P.nr);
    rec(name, y, (long long)r);
    ok &= y == 0 && r == 0;
    if (name == "c_l7qs_o") dump_tiles(name, C.Y, nullptr);
  }
  if (F) CK(cudaFree(F));
  // c_fx1 (co32c): O against cuBLASLt's; its receipts, judged by cgemm_ref.py from the key, differ from c_l7qs's (r_diff)
  if (!C.half) {
    CK(cudaMemsetAsync(C.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(C.R, 0xA5, P.nr * 4, P.st));
    cgemm_run(C, cpick(C, ckern("c_fx1")), C.Y, C.R);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    const unsigned long long y = ecmp(P, C.Y, C.Yr, P.ny), r = ecmp(P, C.R, C.R2, P.nr);
    res += ", \"c_fx1\": {\"y_bad\": " + std::to_string(y) + ", \"r_diff\": " + std::to_string(r) + "}";
    ok &= y == 0 && r > 0;
    dump_tiles("c_fx1", C.Y, C.R);
  }
  // the no-write control: poisoned, nothing run; both compares must fail
  CK(cudaMemsetAsync(C.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(C.R, 0xA5, P.nr * 4, P.st));
  const unsigned long long cy = ecmp(P, C.Y, C.Yr, P.ny), cr = ecmp(P, C.R, C.R2, P.nr);
  const bool control_rejected = cy > 0 && cr > 0;
  rec("control", cy, (long long)cr);
  std::string tj;
  for (int i = 0; i < 3; i++) {
    const int tm = tl[i][0], tn = tl[i][1];
    const std::string sfx = "_" + std::to_string(tm) + "_" + std::to_string(tn) + ".bin";
    dump(dir, ("x" + sfx).c_str(), P.x + (size_t)tm * 128 * k, 128 * (size_t)k * 2, true);
    dump(dir, ("X" + sfx).c_str(), C.X + (size_t)tm * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("PA" + sfx).c_str(), C.PA + (size_t)tm * 128 * 128, 128 * 128, true);
    dump(dir, ("W" + sfx).c_str(), C.W7 + (size_t)tn * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("Wp" + sfx).c_str(), C.Wp + (size_t)tn * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("PB" + sfx).c_str(), C.PB + (size_t)tn * 128 * 128, 128 * 128, true);
    tj += std::string(i ? ", " : "") + "[" + std::to_string(tm) + ", " + std::to_string(tn) + "]";
  }
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"inv_s\": %.9g, \"grid\": %d, \"numerics\": \"%s\", \"key\": [\"%016llx\", \"%016llx\", "
              "\"%016llx\", \"%016llx\"], \"tiles\": [%s], \"kinds\": {\"%s\": \"%s\"}, \"device\": {%s}, \"device_ok\": %s, "
              "\"control_rejected\": %s}\n",
          m, n, k, seed, C.half ? C.inv7 : P.inv_s, P.grid, C.h1 ? "h1" : "co32c", (unsigned long long)s.w[0], (unsigned long long)s.w[1],
          (unsigned long long)s.w[2], (unsigned long long)s.w[3], tj.c_str(), km.c_str(), C.half ? "cohalf" : "co32c", res.c_str(),
          ok ? "true" : "false", control_rejected ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"device_ok\": %s, \"control_rejected\": %s, \"device\": {%s}}\n",
         dir.c_str(), m, n, k, ok ? "true" : "false", control_rejected ? "true" : "false", res.c_str());
  cfree(C);
  return ok && control_rejected ? 0 : 1;
}

static int cbench(int reps, int m, int n, int k) {
  Nv nv;
  const std::vector<std::string> names = split(getenv("CG_ARMS"), "plain,plainq,e_plains,e_l7qs,c_l7qs,c_l7qs_o,editedq:e_l7qs,cedited:c_l7qs");
  CP C;
  csetup(C, m, n, k, 0x5EED0000u + m + n + k, false);
  std::vector<std::function<void(CP&)>> arms;
  for (auto& a : names) arms.push_back(carm(a));
  const int NA = (int)arms.size();
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_arm = [&](int j, int calls) {
    CK(cudaEventRecord(e0, C.E.st));
    for (int c = 0; c < calls; c++) arms[j](C);
    CK(cudaEventRecord(e1, C.E.st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return (double)ms / calls;
  };
  std::vector<int> calls(NA);
  for (int j = 0; j < NA; j++) {
    time_arm(j, 1);
    const double t = time_arm(j, 2);
    calls[j] = std::max(2, std::min(400, (int)(20.0 / std::max(t, 1e-4))));
  }
  CK(cudaGetLastError());
  std::vector<std::vector<double>> ms(NA), mhz(NA);
  for (int r = 0; r < reps; r++)
    for (int j = 0; j < NA; j++) {
      const int a = (j + r) % NA;
      ms[a].push_back(time_arm(a, calls[a]));
      mhz[a].push_back(nv.sm());
    }
  CK(cudaGetLastError());
  printf("{\"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d, \"lt_algo\": %d, \"arms\": {", m, n, k, C.E.grid, C.E.plain.algo_id);
  for (int j = 0; j < NA; j++) {
    printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", names[j].c_str(), median(ms[j]),
           median(mhz[j]), calls[j]);
    for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
    printf("], \"mhz_all\": [");
    for (size_t i = 0; i < mhz[j].size(); i++) printf("%s%.0f", i ? ", " : "", mhz[j][i]);
    printf("]}");
  }
  printf("}}\n");
  fflush(stdout);
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  cfree(C);
  return 0;
}

static int cone(const std::string& arm, int m, int n, int k) {
  CP C;
  csetup(C, m, n, k, 0x5EED0000u + m + n + k, false);
  auto f = carm(arm);
  for (int i = 0; i < 3; i++) f(C);
  CK(cudaStreamSynchronize(C.E.st));
  CK(cudaProfilerStart());
  f(C);
  CK(cudaStreamSynchronize(C.E.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"one\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d}\n", arm.c_str(), m, n, k, C.E.grid);
  cfree(C);
  return 0;
}

#ifndef CGEMM_LIB
int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return cgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 6 && !strcmp(argv[1], "bench")) return cbench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 6 && !strcmp(argv[1], "one")) return cone(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: cgemm gate DIR M N K SEED | bench REPS M N K | one ARM M N K\n");
  return 2;
}
#endif
