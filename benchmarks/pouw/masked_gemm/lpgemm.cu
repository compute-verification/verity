// One-side-peeled noise on sm_120a. For Y = A B^T
// (A m x k, B n x k, problems stacked in rows as fgemm.cu's Args batch): A' = A + U W^T with U (m x 32, {±1..±8}, shared by
// the batch) and W stacked identities, so A'_ik = A_ik + U_{i, k mod 32}; B' = B + F with F fresh ±1 (sha_noise.cuh's pm1,
// B's tag); the prover commits the stride-32 prefix sums of A' B'^T and serves A B'^T = A' B'^T - U S^T, S = W^T B'^T
// (S[j, l] = sum_t B'[j, 32 t + l], one n x 32 a problem). U is SHA-512 in counter mode, tag 6: block j = rows 4j..4j+3,
// entry e = (row % 4) 32 + l digest nibble e (byte e / 2, the low nibble for even e), u = (v & 7) + 1, negated if v & 8.
// The peel on the tensor cores, two ways:
//   lt: cuBLASLt on operands extended by X = 32 (mx + 1) columns, A's -U then mx blocks of -15 U, B's S's digits
//       b0 = S - 15 Sh in [-7, 7] (Sh = floor((S + 7) / 15)) and Sh split into mx int8 parts, so one int8 GEMM at k + X
//       returns A B'^T (mx = ceil(max |Sh| / 127), |B'| <= 120)
//   tail: fgemm.cu's tail stage, P_A = [U U U U] (shared by the batch) against P_B = [S_h1 | S_h2 | S_l | 0]
//       (S = 256 (S_h1 + S_h2) + S_l), so the peel's acc -= 256 U S_h^T + U S_l^T is one extra stage of 128 at the end;
//       A' formed by a pass (t), added in shared memory from U's row (f serialized, o overlapped; fgemm.cu's NS -2), or
//       added to A's fragments in registers after each ldmatrix (r; NS -3: U's period, 32 columns, is one atom)
// B's side: the edit pass then a pass for S's digits (bsep), or one pass doing both (bfus: a warp a row group of 4 rows,
// its lanes hashing 32 patches at once; at k = 128 a thread a patch); U's hashes run in the first blocks of B's pass.
// Receipts: duo at L = 32 over the first k columns (td, fd, od, rd), the tail outside.
// Reference (bonly): F's pass and cuBLASLt on A and B' at k, no U and no peel: the served product without A' B'^T, which only
// the receipts need.
// A and B in [-119, 119], so A' and B' can't saturate and every product is exact in s32.
//   lpgemm gate DIR M N K BATCH SEED GRID   every arm at one small shape, inputs and outputs to DIR for lpref.py
//   lpgemm bench REPS SHAPE...              interleaved timing, one JSON line a shape (SHAPE = name:m:n:k:batch); LP_ARMS
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a lpgemm.cu -lcuda -lcublasLt -lnvidia-ml
// -DLP_UW=64: U in {±1..±64} (u_byte's comment), A then in [-63, 63]; -DLP_LIB: no main (benchmarks/e2e_slice/slice.cu);
// -DLP_NO_ROUTES: no routes either, so bsp's, bsp1's and speel's divisions (which the SASS gate refuses) aren't built
// (benchmarks/pouw/edited_gemm/egemm.cu)
#define MG_LIB
#include "mgemm.cu"
#include "sha_pass.cuh"

constexpr uint32_t U_TAG = 6;
constexpr int F_TAG = 1;   // pm1, B's side
// B's noise word from F's pm1 word: F, or with -DLP_DITHER the dither b = (1 - F) / 2 in {0, 1} from the same coin
// bits (B' = B + b, the served X = floor(x / s) + b)
#ifdef LP_DITHER
DEV uint32_t fb_word(uint32_t w) { return (w >> 1) & 0x01010101u; }
#else
DEV uint32_t fb_word(uint32_t w) { return w; }
#endif

#define GEMML(NAME, RS, KIND, NZ, NS)                                                                                  \
  __global__ void __launch_bounds__(ML<128, 128, 32, 3, false, NZ, NS>::NT, 1) NAME(const __grid_constant__ Args a) { \
    gemm_body<128, 128, 32, 3, true, RS, KIND, false, NZ, NS>(a);                                                      \
  }
GEMML(t_plain, 0, NONE, 0, 0)
GEMML(t_duo32, 32, DUO, 0, 0)
GEMML(t_duos32, 32, DUOS, 0, 0)
GEMML(f_plain, 0, NONE, 1, -2)
GEMML(f_duo32, 32, DUO, 1, -2)
GEMML(o_plain, 0, NONE, 2, -2)
GEMML(o_duo32, 32, DUO, 2, -2)
GEMML(r_plain, 0, NONE, 0, -3)
GEMML(r_duo32, 32, DUO, 0, -3)
static_assert(ML<128, 128, 32, 3, false, 2, -2>::SMEM == CSMEM, "one shared-memory size");

// LP_UW 64: U in {±1..±64} (A in [-63, 63]), one digest byte an entry: block j = rows 2j, 2j + 1, entry e = (row % 2) 32 + l
// digest byte e, u = (v & 63) + 1, negated if v & 64 (bit 7 unread)
#ifndef LP_UW
#define LP_UW 8
#endif
static_assert(LP_UW == 8 || LP_UW == 64, "U's magnitude bound: 8 or 64");
DEV uint32_t u_byte(uint32_t v) {
  const uint32_t u = (v & (LP_UW - 1u)) + 1u;
  return (v & (uint32_t)LP_UW ? 0u - u : u) & 0xFFu;
}
// U's rows 4j..4j+3 into P_A (m x 128, a row [U U U U]). B's passes run it in their first blocks, so U's few
// hashes hide under B's pass; gen_u is it alone
DEV void gen_u_blk(uint8_t* __restrict__ PA, uint32_t j, uint32_t blocks, const Sig& sg) {
  if (j >= blocks) return;
#if LP_UW == 64
#pragma unroll
  for (int h = 0; h < 2; h++) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)U_TAG << 56 | (2 * j + h), st);
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
      uint32_t w[8];
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t v = 0;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          const int e = 32 * rr + 4 * q + t;
          v |= u_byte((uint32_t)(st[e / 8] >> (56 - 8 * (e % 8))) & 0xFFu) << (8 * t);
        }
        w[q] = v;
      }
      uint4* row = reinterpret_cast<uint4*>(PA + (size_t)(4 * j + 2 * h + rr) * 128);
#pragma unroll
      for (int c = 0; c < 8; c++) row[c] = make_uint4(w[4 * (c & 1)], w[4 * (c & 1) + 1], w[4 * (c & 1) + 2], w[4 * (c & 1) + 3]);
    }
  }
#else
  uint64_t st[8];
  sha_block(sg.w, (uint64_t)U_TAG << 56 | j, st);
#pragma unroll
  for (int r = 0; r < 4; r++) {
    uint32_t w[8];
#pragma unroll
    for (int q = 0; q < 8; q++) {   // entries 32 r + 4 q .. + 3: digest bytes 16 r + 2 q (high) and + 1 (low)
      const uint32_t b16 = (uint32_t)(st[2 * r + q / 4] >> (48 - 16 * (q % 4))) & 0xFFFFu, b0 = b16 >> 8, b1 = b16 & 0xFFu;
      w[q] = u_byte(b0 & 15) | u_byte(b0 >> 4) << 8 | u_byte(b1 & 15) << 16 | u_byte(b1 >> 4) << 24;
    }
    uint4* row = reinterpret_cast<uint4*>(PA + (size_t)(4 * j + r) * 128);
#pragma unroll
    for (int h = 0; h < 8; h++) row[h] = make_uint4(w[4 * (h & 1)], w[4 * (h & 1) + 1], w[4 * (h & 1) + 2], w[4 * (h & 1) + 3]);
  }
#endif
}
__global__ void __launch_bounds__(256) gen_u(uint8_t* __restrict__ PA, uint32_t blocks, const __grid_constant__ Sig sg) {
  gen_u_blk(PA, blockIdx.x * blockDim.x + threadIdx.x, blocks, sg);
}
// B's passes: grid blocks 0..ub-1 generate U (ublocks of 4 rows), the rest do the pass
struct UJob { uint8_t* PA; uint32_t ub, ublocks; };
DEV bool u_part(const UJob& u, const Sig& sg) {
  if (blockIdx.x >= u.ub) return false;
  gen_u_blk(u.PA, blockIdx.x * blockDim.x + threadIdx.x, u.ublocks, sg);
  return true;
}

// XA (rows x ld) = A + U in the first K columns; EXT also columns K + 32 c + l: -U[., l] (c = 0), -15 U[., l] (c = 1..mx)
// (a 16 x 16 block: 16 chunks of 16 rows; chunk x of row y)
template <bool EXT>
__global__ void __launch_bounds__(256) edit_a(const uint8_t* __restrict__ A, const uint8_t* __restrict__ PA, uint8_t* __restrict__ XA,
                                              int rows, int K, int M, int ld, int cw) {
  const int c = blockIdx.x * 16 + threadIdx.x, ri = blockIdx.y * 16 + threadIdx.y;
  if (c < cw && ri < rows) {
    const size_t r = (size_t)ri;
    const int col = 16 * c;
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(PA + (size_t)(ri % M) * 128 + (col & 31)));
    uint4 o;
    if (!EXT || col < K) {
      const uint4 x = *reinterpret_cast<const uint4*>(A + r * K + col);
      o = make_uint4(add_noise<false>(x.x, u.x), add_noise<false>(x.y, u.y), add_noise<false>(x.z, u.z), add_noise<false>(x.w, u.w));
    } else {
      const int f = (col - K) < 32 ? -1 : -15;
      uint32_t ow[4];
      const uint32_t uw[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) v |= ((uint32_t)(f * (int)(int8_t)(uw[q] >> (8 * b))) & 0xFFu) << (8 * b);
        ow[q] = v;
      }
      o = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
    *reinterpret_cast<uint4*>(XA + r * ld + col) = o;
  }
}

// B' = B + F into XB (row stride ld), one thread a patch
__global__ void __launch_bounds__(256) edit_b(const uint8_t* __restrict__ B, uint8_t* __restrict__ XB, uint32_t pb, uint32_t KB, int ld,
                                              const UJob uj, const __grid_constant__ Sig sg) {
  if (u_part(uj, sg)) return;
  const uint32_t p = (blockIdx.x - uj.ub) * blockDim.x + threadIdx.x;
  if (p >= pb) return;
  const size_t k = (size_t)KB * 128, row0 = (size_t)(p / KB) * 4, col0 = (size_t)(p % KB) * 128;
  sha_patch<1>(sg.w, F_TAG, p, [&](int c, const uint32_t(&n)[4]) {
    const size_t r = row0 + c / 8, col = col0 + (c % 8) * 16;
    const uint4 x = *reinterpret_cast<const uint4*>(B + r * k + col);
    *reinterpret_cast<uint4*>(XB + r * ld + col) =
        make_uint4(add_noise<false>(x.x, fb_word(n[0])), add_noise<false>(x.y, fb_word(n[1])), add_noise<false>(x.z, fb_word(n[2])),
                   add_noise<false>(x.w, fb_word(n[3])));
  });
}

DEV void acc16(int (&s)[16], const uint4 v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; q++)
#pragma unroll
    for (int b = 0; b < 4; b++) s[4 * q + b] += (int)(int8_t)(w[q] >> (8 * b));
}
DEV uint4 pack16(const int (&v)[16]) {
  uint32_t w[4];
#pragma unroll
  for (int q = 0; q < 4; q++)
    w[q] = ((uint32_t)v[4 * q] & 0xFFu) | ((uint32_t)v[4 * q + 1] & 0xFFu) << 8 | ((uint32_t)v[4 * q + 2] & 0xFFu) << 16 |
           ((uint32_t)v[4 * q + 3] & 0xFFu) << 24;
  return make_uint4(w[0], w[1], w[2], w[3]);
}
// floor(a / b) for |a| < 2^20 by b's reciprocal, corrected by one either way
DEV int fdiv_r(int a, int b, float rb) {
  int q = __float2int_rd((float)a * rb);
  q -= q * b > a;
  q += (q + 1) * b <= a;
  return q;
}
// S[row, 16 h + e] = s[e]'s digits: P_B's atoms a = j, j + nj, .. < 4 (S_h1, S_h2, S_l, 0) and, EXT, XB's extension blocks
// c = j, j + nj, .. <= mx (c = 0: b0; c = 1..mx: Sh's parts, q + [c > mx - r] for Sh = q mx + r)
template <bool EXT>
DEV void digits(const int (&s)[16], size_t row, int h, int j, int nj, int8_t* PB, uint8_t* XB, int K, int ld, int mx) {
  for (int a = j; a < 4; a += nj) {
    int v[16];
#pragma unroll
    for (int e = 0; e < 16; e++) {
      const int lo = ((s[e] + 128) & 255) - 128, hi = (s[e] - lo) >> 8, h1 = hi >> 1;
      v[e] = a == 0 ? h1 : a == 1 ? hi - h1 : a == 2 ? lo : 0;
    }
    *reinterpret_cast<uint4*>(PB + row * 128 + 32 * a + 16 * h) = pack16(v);
  }
  if constexpr (EXT) {
    if (j > mx) return;
    const float rm = 1.0f / (float)mx;
    int b0[16], q[16], r[16];
#pragma unroll
    for (int e = 0; e < 16; e++) {
      const int sh = fdiv(s[e] + 7, 15);
      b0[e] = s[e] - 15 * sh;
      q[e] = fdiv_r(sh, mx, rm);
      r[e] = sh - q[e] * mx;
    }
    for (int c = j; c <= mx; c += nj) {
      int v[16];
#pragma unroll
      for (int e = 0; e < 16; e++) v[e] = c == 0 ? b0[e] : q[e] + (c > mx - r[e] ? 1 : 0);
      *reinterpret_cast<uint4*>(XB + row * ld + K + 32 * c + 16 * h) = pack16(v);
    }
  }
}
// S from XB's first K columns, one warp a row: lane L sums chunks c = L (mod 32), so columns 16 (L & 1) + e mod 32
template <bool EXT>
__global__ void __launch_bounds__(256) speel(uint8_t* __restrict__ XB, int8_t* __restrict__ PB, int rows, int K, int ld, int mx) {
  const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const uint8_t* x = XB + (size_t)row * ld;
  int s[16] = {0};
#pragma unroll 4
  for (int c = lane; c < K / 16; c += 32) acc16(s, *reinterpret_cast<const uint4*>(x + 16 * c));
#pragma unroll
  for (int e = 0; e < 16; e++)
#pragma unroll
    for (int o = 2; o < 32; o <<= 1) s[e] += __shfl_xor_sync(0xFFFFFFFFu, s[e], o);
  digits<EXT>(s, row, lane & 1, lane >> 1, 16, PB, XB, K, ld, mx);
}
// the same for short rows (K <= 512), one thread a half row (columns 16 h .. mod 32)
template <bool EXT>
__global__ void __launch_bounds__(256) speel_t(uint8_t* __restrict__ XB, int8_t* __restrict__ PB, int rows, int K, int ld, int mx) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x, row = t >> 1, h = t & 1;
  if (row >= rows) return;
  const uint8_t* x = XB + (size_t)row * ld;
  int s[16] = {0};
#pragma unroll 4
  for (int c = h; c < K / 16; c += 2) acc16(s, *reinterpret_cast<const uint4*>(x + 16 * c));
  digits<EXT>(s, row, h, 0, 1, PB, XB, K, ld, mx);
}
// B' = B + F and S's digits in one pass, one warp a row group (4 rows, patches g KB ..): its lanes hash 32 patches at once
// into shared memory, then lane i edits chunk i of each (row i / 8, columns 16 (i % 8) ..) and sums its 16 columns
template <bool EXT>
__global__ void __launch_bounds__(256) bsp(const uint8_t* __restrict__ B, uint8_t* __restrict__ XB, int8_t* __restrict__ PB, int groups,
                                           int K, int ld, int mx, const UJob uj, const __grid_constant__ Sig sg) {
  __shared__ uint64_t dg[8][32][8];
  if (u_part(uj, sg)) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = (blockIdx.x - uj.ub) * 8 + warp;
  if (g >= groups) return;
  const int KB = K / 128, r = lane >> 3, cc = lane & 7;
  const size_t row = (size_t)g * 4 + r;
  const uint8_t* src = B + row * K + cc * 16;
  uint8_t* dst = XB + row * ld + cc * 16;
  int s[16] = {0};
  for (int t0 = 0; t0 < KB; t0 += 32) {
    const int np = min(32, KB - t0);
    if (lane < np) {
      uint64_t st[8];
      sha_block(sg.w, (uint64_t)F_TAG << 56 | ((uint64_t)g * KB + t0 + lane), st);
#pragma unroll
      for (int w = 0; w < 8; w++) dg[warp][lane][w] = st[w];
    }
    __syncwarp();
#pragma unroll 4
    for (int t = 0; t < np; t++) {
      const uint32_t b16 = (uint32_t)(dg[warp][t][lane / 4] >> (48 - 16 * (lane % 4))) & 0xFFFFu;
      const uint32_t bits = (b16 >> 8) | ((b16 & 0xFF) << 8);
      const size_t off = (size_t)(t0 + t) * 128;
      const uint4 x = *reinterpret_cast<const uint4*>(src + off);
      const uint4 y = make_uint4(add_noise<false>(x.x, fb_word(pm1_word(bits & 0xF))),
                                 add_noise<false>(x.y, fb_word(pm1_word((bits >> 4) & 0xF))),
                                 add_noise<false>(x.z, fb_word(pm1_word((bits >> 8) & 0xF))),
                                 add_noise<false>(x.w, fb_word(pm1_word(bits >> 12))));
      *reinterpret_cast<uint4*>(dst + off) = y;
      acc16(s, y);
    }
    __syncwarp();
  }
#pragma unroll
  for (int e = 0; e < 16; e++) {
    s[e] += __shfl_xor_sync(0xFFFFFFFFu, s[e], 2);
    s[e] += __shfl_xor_sync(0xFFFFFFFFu, s[e], 4);
  }
  digits<EXT>(s, row, cc & 1, cc >> 1, 4, PB, XB, K, ld, mx);
}
// the same at K = 128 (a patch is 4 whole rows), one thread a patch, its rows' sums finished one row at a time
template <bool EXT>
__global__ void __launch_bounds__(256) bsp1(const uint8_t* __restrict__ B, uint8_t* __restrict__ XB, int8_t* __restrict__ PB,
                                            uint32_t groups, int ld, int mx, const UJob uj, const __grid_constant__ Sig sg) {
  if (u_part(uj, sg)) return;
  const uint32_t p = (blockIdx.x - uj.ub) * blockDim.x + threadIdx.x;
  if (p >= groups) return;
  int s0[16] = {0}, s1[16] = {0};
  sha_patch<1>(sg.w, F_TAG, p, [&](int c, const uint32_t(&n)[4]) {
    const size_t row = (size_t)p * 4 + c / 8;
    if (c % 8 == 0) {
#pragma unroll
      for (int e = 0; e < 16; e++) s0[e] = s1[e] = 0;
    }
    const uint4 x = *reinterpret_cast<const uint4*>(B + row * 128 + (c % 8) * 16);
    const uint4 y = make_uint4(add_noise<false>(x.x, fb_word(n[0])), add_noise<false>(x.y, fb_word(n[1])), add_noise<false>(x.z, fb_word(n[2])),
                               add_noise<false>(x.w, fb_word(n[3])));
    *reinterpret_cast<uint4*>(XB + row * ld + (c % 8) * 16) = y;
    if (c % 2 == 0) acc16(s0, y); else acc16(s1, y);
    if (c % 8 == 7) {
      digits<EXT>(s0, row, 0, 0, 1, PB, XB, 128, ld, mx);
      digits<EXT>(s1, row, 1, 0, 1, PB, XB, 128, ld, mx);
    }
  });
}

static CUtensorMap tmap_ld(const void* p, int depth, int ld, int rows) {
  CUtensorMap m;
  const cuuint64_t dims[2] = {(cuuint64_t)depth, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)ld};
  const cuuint32_t box[2] = {128, 128}, es[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(p), dims, strides, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled: %d\n", int(r)); exit(3); }
  return m;
}

static int fdiv_h(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
// the extension's parts: |S| <= 120 k / 32, Sh = floor((S + 7) / 15), mx = ceil(max |Sh| / 127)
static int ext_parts(int k) {
  const int smax = 120 * (k / 32), sh = std::max(fdiv_h(smax + 7, 15), -fdiv_h(-smax + 7, 15));
  return (sh + 126) / 127;
}

#ifndef LP_NO_ROUTES
struct LProb {
  int m, n, k, batch, grid, Kp, mx;
  uint32_t pb, KB;
  uint8_t *A, *B, *XA, *XB, *PA;
  int8_t* PB;
  int* U;
  uint32_t *R, *G;
  size_t na, nb, nxa, nxb, ub, rb;
  LtG plain, ltk, ltb;
  Maps *mAB, *mT, *mF;
  uint32_t seed, inst = 0;
  cudaStream_t st;
};

static void lsetup(LProb& P, int m, int n, int k, int batch, uint32_t seed, int grid_cap, bool gate) {
  P.m = m; P.n = n; P.k = k; P.batch = batch; P.seed = seed;
  if (m % 128 || n % 128 || k % 128) { fprintf(stderr, "shape %d %d %d not tiled\n", m, n, k); exit(2); }
  P.mx = ext_parts(k);
  P.Kp = k + 32 * (P.mx + 1);
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  P.grid = std::min((m / 128) * (n / 128) * batch, grid_cap > 0 ? grid_cap : nsm);
  CK(cudaStreamCreate(&P.st));
  P.na = (size_t)batch * m * k; P.nb = (size_t)batch * n * k;
  P.nxa = (size_t)batch * m * P.Kp; P.nxb = (size_t)batch * n * P.Kp;
  if (P.nb / 512 >= (1ull << 32)) { fprintf(stderr, "operand past the counter\n"); exit(2); }
  P.KB = k / 128; P.pb = (uint32_t)(P.nb / 512);
  CK(cudaMalloc(&P.A, P.na)); CK(cudaMalloc(&P.B, P.nb));
  CK(cudaMalloc(&P.XA, P.nxa)); CK(cudaMalloc(&P.XB, P.nxb));
  CK(cudaMalloc(&P.PA, (size_t)m * 128)); CK(cudaMalloc(&P.PB, (size_t)batch * n * 128));
  CK(cudaMalloc(&P.G, (size_t)P.pb * 4));
  P.ub = (size_t)batch * m * n * 4;
  P.rb = (size_t)(m / 128) * (n / 128) * batch * CNT * 8 * 4;
  CK(cudaMalloc(&P.U, P.ub)); CK(cudaMalloc(&P.R, P.rb));
  sfill<I8><<<4096, 256, 0, P.st>>>(P.A, P.na, mix32(seed ^ 0x1111u));
  sfill<I8><<<4096, 256, 0, P.st>>>(P.B, P.nb, mix32(seed ^ 0x2222u));
  if (gate)   // rows 0 and 1 of each problem at +119 and -119, so A' and S reach their ranges' ends
    for (int b = 0; b < batch; b++) {
      CK(cudaMemsetAsync(P.A + (size_t)b * m * k, 0x77, k, P.st)); CK(cudaMemsetAsync(P.A + ((size_t)b * m + 1) * k, 0x89, k, P.st));
      CK(cudaMemsetAsync(P.B + (size_t)b * n * k, 0x77, k, P.st)); CK(cudaMemsetAsync(P.B + ((size_t)b * n + 1) * k, 0x89, k, P.st));
    }
  CK(cudaStreamSynchronize(P.st));
  P.plain = ltg(I8, m, n, k, k, k, n, P.A, P.B, P.U, 0, P.st, batch);
  P.ltk = ltg(I8, m, n, P.Kp, P.Kp, P.Kp, n, P.XA, P.XB, P.U, 0, P.st, batch);
  P.ltb = ltg(I8, m, n, k, k, P.Kp, n, P.A, P.XB, P.U, 0, P.st, batch);
  const CUtensorMap pa = tmap(P.PA, 128, m, 128), pbm = tmap(P.PB, 128, batch * n, 128);
  P.mAB = upload_maps(Maps{tmap(P.A, k, batch * m, 128), tmap(P.B, k, batch * n, 128), tmap(P.A, k, batch * m, 128), tmap(P.B, k, batch * n, 128)});
  P.mT = upload_maps(Maps{tmap_ld(P.XA, k, P.Kp, batch * m), tmap_ld(P.XB, k, P.Kp, batch * n), pa, pbm});
  P.mF = upload_maps(Maps{tmap(P.A, k, batch * m, 128), tmap_ld(P.XB, k, P.Kp, batch * n), pa, pbm});
  for (Kern kern : {RCK[0][0], RCK[0][1], (Kern)t_plain, (Kern)t_duo32, (Kern)t_duos32, (Kern)f_plain, (Kern)f_duo32,
                     (Kern)o_plain, (Kern)o_duo32, (Kern)r_plain, (Kern)r_duo32})
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
}
static void lfree(LProb& P) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)P.A, (void*)P.B, (void*)P.XA, (void*)P.XB, (void*)P.PA, (void*)P.PB, (void*)P.U, (void*)P.R, (void*)P.G,
                  (void*)P.mAB, (void*)P.mT, (void*)P.mF})
    CK(cudaFree(p));
  CK(cudaStreamDestroy(P.st));
}

enum { L_PLAIN, L_CPLAIN, L_CDUO, L_GENU, L_GENF, L_FORMA, L_FORMAX, L_FORMB, L_SPEEL, L_SPEELX, L_BSP, L_BSPX, L_LTK, L_TK, L_FK,
       L_OK, L_RK, L_BONLY, L_ROUTE };
enum { R_LT, R_T, R_F, R_O, R_R };
static const char* RN[5] = {"lt", "t", "f", "o", "r"};
struct LArm { std::string name; int kind, route, bfus, duo; };
static std::vector<LArm> lall() {
  std::vector<LArm> v{{"plain", L_PLAIN}, {"cplain", L_CPLAIN}, {"cduo", L_CDUO}, {"genU", L_GENU}, {"genF", L_GENF},
                      {"formA", L_FORMA}, {"formAx", L_FORMAX}, {"formB", L_FORMB}, {"speel", L_SPEEL}, {"speelx", L_SPEELX},
                      {"bsp", L_BSP}, {"bspx", L_BSPX}, {"ltK", L_LTK}, {"tK", L_TK}, {"fK", L_FK}, {"oK", L_OK}, {"rK", L_RK},
                      {"bonly", L_BONLY}};
  for (int r = 0; r < 5; r++)
    for (int d = 0; d < (r == R_LT ? 1 : 2); d++)
      for (int bf = 0; bf < 2; bf++) v.push_back({std::string(RN[r]) + (d ? "d" : "") + (bf ? "_bfus" : "_bsep"), L_ROUTE, r, bf, d});
  return v;
}
// the components the routes run, each checked through the route named
static const std::pair<const char*, const char*> COVER[] = {{"formA", "t_bsep"}, {"formAx", "lt_bsep"}, {"formB", "lt_bsep"},
                                                            {"speel", "t_bsep"}, {"speelx", "lt_bsep"}, {"bsp", "f_bfus"},
                                                            {"bspx", "lt_bfus"}, {"ltK", "lt_bsep"}, {"tK", "t_bsep"},
                                                            {"fK", "f_bsep"}, {"oK", "o_bsep"}, {"rK", "r_bsep"}};
static bool larm_ok(const LProb& P, const LArm& a) {
  if (a.kind == L_PLAIN) return P.plain.algo_id != -2;
  if (a.kind == L_LTK || (a.kind == L_ROUTE && a.route == R_LT)) return P.ltk.algo_id != -2;
  if (a.kind == L_BONLY) return P.ltb.algo_id != -2;
  return true;
}
static UJob ujob(LProb& P) { return UJob{P.PA, (uint32_t)(P.m / 4 + 255) / 256, (uint32_t)(P.m / 4)}; }
static void genU(LProb& P, const Sig& s) { gen_u<<<(P.m / 4 + 255) / 256, 256, 0, P.st>>>(P.PA, P.m / 4, s); }
static void formA(LProb& P, bool ext) {
  const int cw = (ext ? P.Kp : P.k) / 16, rows = P.batch * P.m;
  const dim3 g((cw + 15) / 16, (rows + 15) / 16), b(16, 16);
  if (ext) edit_a<true><<<g, b, 0, P.st>>>(P.A, P.PA, P.XA, rows, P.k, P.m, P.Kp, cw);
  else edit_a<false><<<g, b, 0, P.st>>>(P.A, P.PA, P.XA, rows, P.k, P.m, P.Kp, cw);
}
// B's passes, U's blocks first in each
static void formB(LProb& P, const Sig& s) {
  const UJob u = ujob(P);
  edit_b<<<u.ub + (P.pb + 255) / 256, 256, 0, P.st>>>(P.B, P.XB, P.pb, P.KB, P.Kp, u, s);
}
static void speelS(LProb& P, bool ext) {
  const int rows = P.batch * P.n;
  if (P.k <= 512) {
    if (ext) speel_t<true><<<(2 * rows + 255) / 256, 256, 0, P.st>>>(P.XB, P.PB, rows, P.k, P.Kp, P.mx);
    else speel_t<false><<<(2 * rows + 255) / 256, 256, 0, P.st>>>(P.XB, P.PB, rows, P.k, P.Kp, P.mx);
  } else {
    if (ext) speel<true><<<(rows + 7) / 8, 256, 0, P.st>>>(P.XB, P.PB, rows, P.k, P.Kp, P.mx);
    else speel<false><<<(rows + 7) / 8, 256, 0, P.st>>>(P.XB, P.PB, rows, P.k, P.Kp, P.mx);
  }
}
static void bspS(LProb& P, bool ext, const Sig& s) {
  const int groups = P.batch * P.n / 4;
  const UJob u = ujob(P);
  if (P.k == 128) {
    const unsigned g = u.ub + (groups + 255) / 256;
    if (ext) bsp1<true><<<g, 256, 0, P.st>>>(P.B, P.XB, P.PB, groups, P.Kp, P.mx, u, s);
    else bsp1<false><<<g, 256, 0, P.st>>>(P.B, P.XB, P.PB, groups, P.Kp, P.mx, u, s);
  } else {
    const unsigned g = u.ub + (groups + 7) / 8;
    if (ext) bsp<true><<<g, 256, 0, P.st>>>(P.B, P.XB, P.PB, groups, P.k, P.Kp, P.mx, u, s);
    else bsp<false><<<g, 256, 0, P.st>>>(P.B, P.XB, P.PB, groups, P.k, P.Kp, P.mx, u, s);
  }
}
static void bside(LProb& P, bool fused, bool ext, const Sig& s) {
  if (fused) bspS(P, ext, s);
  else { formB(P, s); speelS(P, ext); }
}
static void lgemm(LProb& P, Kern kern, const Maps* maps, const Sig& s) {
  Args a{maps, P.U, P.R, P.m, P.n, P.k, 8, P.seed, 0, P.batch, 0, 0, {s.w[0], s.w[1], s.w[2], s.w[3]}, P.PA, P.m};
  kern<<<P.grid, CNT, CSMEM, P.st>>>(a);
}
static void lrun(LProb& P, const LArm& a) {
  const Sig s = sigma(P.seed, P.inst++);
  switch (a.kind) {
    case L_PLAIN: ltg_run(P.plain, P.st); break;
    case L_CPLAIN: lgemm(P, RCK[0][0], P.mAB, s); break;
    case L_CDUO: lgemm(P, RCK[0][1], P.mAB, s); break;
    case L_GENU: genU(P, s); break;
    case L_GENF: sha_gen<1><<<(P.pb + 255) / 256, 256, 0, P.st>>>(P.G, 0, P.pb, s); break;
    case L_FORMA: formA(P, false); break;
    case L_FORMAX: formA(P, true); break;
    case L_FORMB: formB(P, s); break;
    case L_SPEEL: speelS(P, false); break;
    case L_SPEELX: speelS(P, true); break;
    case L_BSP: bspS(P, false, s); break;
    case L_BSPX: bspS(P, true, s); break;
    case L_LTK: ltg_run(P.ltk, P.st); break;
    case L_TK: lgemm(P, t_plain, P.mT, s); break;
    case L_FK: lgemm(P, f_plain, P.mF, s); break;
    case L_OK: lgemm(P, o_plain, P.mF, s); break;
    case L_RK: lgemm(P, r_plain, P.mF, s); break;
    case L_BONLY:
      edit_b<<<(P.pb + 255) / 256, 256, 0, P.st>>>(P.B, P.XB, P.pb, P.KB, P.Kp, UJob{P.PA, 0, 0}, s);
      ltg_run(P.ltb, P.st);
      break;
    case L_ROUTE:
      bside(P, a.bfus, a.route == R_LT, s);
      if (a.route == R_LT) {
        formA(P, true);
        ltg_run(P.ltk, P.st);
      } else if (a.route == R_T) {
        formA(P, false);
        lgemm(P, a.duo ? t_duo32 : t_plain, P.mT, s);
      } else if (a.route == R_F) {
        lgemm(P, a.duo ? f_duo32 : f_plain, P.mF, s);
      } else if (a.route == R_O) {
        lgemm(P, a.duo ? o_duo32 : o_plain, P.mF, s);
      } else {
        lgemm(P, a.duo ? r_duo32 : r_plain, P.mF, s);
      }
      break;
  }
}

static int lgate(const std::string& dir, int m, int n, int k, int batch, uint32_t seed, int grid) {
  LProb P;
  lsetup(P, m, n, k, batch, seed, grid, true);
  dump(dir, "A.bin", P.A, P.na, true); dump(dir, "B.bin", P.B, P.nb, true);
  std::string arms_json, skipped, cover;
  for (const LArm& a : lall()) {
    bool component = false;
    for (auto& [c, by] : COVER) component |= a.name == c;
    if (component) continue;
    if (!larm_ok(P, a)) { skipped += (skipped.empty() ? "\"" : ", \"") + a.name + "\""; continue; }
    CK(cudaMemset(P.U, 0xA5, P.ub)); CK(cudaMemset(P.R, 0xA5, P.rb)); CK(cudaMemset(P.XA, 0xA5, P.nxa));
    CK(cudaMemset(P.XB, 0xA5, P.nxb)); CK(cudaMemset(P.PA, 0xA5, (size_t)m * 128)); CK(cudaMemset(P.PB, 0xA5, (size_t)batch * n * 128));
    CK(cudaMemset(P.G, 0xA5, (size_t)P.pb * 4));
    P.inst = 0;
    lrun(P, a);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    if (a.kind == L_GENU) {
      dump(dir, "PA_genU.bin", P.PA, (size_t)m * 128, true);
    } else if (a.kind == L_GENF) {
      dump(dir, "G_genF.bin", P.G, (size_t)P.pb * 4, true);
    } else {
      dump(dir, ("U_" + a.name + ".bin").c_str(), P.U, P.ub, true);
      if (a.kind == L_CDUO || a.duo) dump(dir, ("R_" + a.name + ".bin").c_str(), P.R, P.rb, true);
      if (a.kind == L_BONLY) dump(dir, "XB_bonly.bin", P.XB, P.nxb, true);
      if (a.kind == L_ROUTE) {
        dump(dir, ("PA_" + a.name + ".bin").c_str(), P.PA, (size_t)m * 128, true);
        dump(dir, ("XB_" + a.name + ".bin").c_str(), P.XB, P.nxb, true);
        if (a.route != R_LT) dump(dir, ("PB_" + a.name + ".bin").c_str(), P.PB, (size_t)batch * n * 128, true);
        if (a.route == R_LT || a.route == R_T) dump(dir, ("XA_" + a.name + ".bin").c_str(), P.XA, P.nxa, true);
      }
    }
    arms_json += (arms_json.empty() ? "\"" : ", \"") + a.name + "\"";
  }
  for (auto& [c, by] : COVER) cover += std::string(cover.empty() ? "" : ", ") + "\"" + c + "\": \"" + by + "\"";
  const Sig sg = sigma(seed, 0);
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"batch\": %d, \"Kp\": %d, \"mx\": %d, \"seed\": %u, \"grid\": %d, \"TM\": 128, "
              "\"BN\": 128, \"WM\": 32, \"NCT\": %d, \"sigma\": [\"%016llx\", \"%016llx\", \"%016llx\", \"%016llx\"], \"arms\": [%s], "
              "\"skipped\": [%s], \"cover\": {%s}}\n",
          m, n, k, batch, P.Kp, P.mx, seed, P.grid, CNT, (unsigned long long)sg.w[0], (unsigned long long)sg.w[1],
          (unsigned long long)sg.w[2], (unsigned long long)sg.w[3], arms_json.c_str(), skipped.c_str(), cover.c_str());
  fclose(fp);
  printf("gate dump %s: m %d n %d k %d batch %d Kp %d mx %d grid %d\n", dir.c_str(), m, n, k, batch, P.Kp, P.mx, P.grid);
  lfree(P);
  return 0;
}

static int lbench(int reps, const std::vector<std::string>& shapes) {
  Nv nv;
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::vector<LArm> all;
  const char* fa = getenv("LP_ARMS");
  for (const LArm& a : lall()) {
    const std::string want = fa ? std::string(",") + fa + "," : "";
    if (!fa || want.find("," + a.name + ",") != std::string::npos) all.push_back(a);
  }
  for (const auto& sh : shapes) {
    char name[64];
    int m, n, k, batch = 1;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d:%d", name, &m, &n, &k, &batch) < 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    LProb P;
    lsetup(P, m, n, k, batch, 0x5EED0000u + m + n + k + batch, 0, false);
    std::vector<LArm> sel;
    for (const LArm& a : all) if (larm_ok(P, a)) sel.push_back(a);
    const int NA = (int)sel.size();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time_arm = [&](const LArm& a, int calls) {
      CK(cudaEventRecord(e0, P.st));
      for (int c = 0; c < calls; c++) lrun(P, a);
      CK(cudaEventRecord(e1, P.st));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      return (double)ms / calls;
    };
    std::vector<int> calls(NA);
    for (int j = 0; j < NA; j++) {
      time_arm(sel[j], 1);
      const double t = time_arm(sel[j], 2);
      calls[j] = std::max(2, std::min(400, (int)(20.0 / std::max(t, 1e-4))));
    }
    CK(cudaGetLastError());
    std::vector<std::vector<double>> ms(NA), mhz(NA);
    for (int r = 0; r < reps; r++)
      for (int j = 0; j < NA; j++) {
        const int a = (j + r) % NA;
        ms[a].push_back(time_arm(sel[a], calls[a]));
        mhz[a].push_back(nv.sm());
      }
    std::string skipped;
    for (const LArm& a : all) if (!larm_ok(P, a)) skipped += (skipped.empty() ? "\"" : ", \"") + a.name + "\"";
    printf("{\"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"batch\": %d, \"Kp\": %d, \"mx\": %d, \"sms\": %d, \"gpu\": \"%s\", "
           "\"grid\": %d, \"skipped\": [%s], \"lt_algo\": [%d, %d], \"arms\": {",
           name, m, n, k, batch, P.Kp, P.mx, nsm, prop.name, P.grid, skipped.c_str(), P.plain.algo_id, P.ltk.algo_id);
    for (int j = 0; j < NA; j++) {
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", sel[j].name.c_str(), median(ms[j]),
             median(mhz[j]), calls[j]);
      for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
      printf("]}");
    }
    printf("}}\n");
    fflush(stdout);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    lfree(P);
  }
  return 0;
}
#endif

#ifndef LP_LIB
int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 9 && !strcmp(argv[1], "gate"))
    return lgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), (uint32_t)strtoul(argv[7], 0, 0), atoi(argv[8]));
  if (argc >= 4 && !strcmp(argv[1], "bench")) {
    std::vector<std::string> s(argv + 3, argv + argc);
    return lbench(atoi(argv[2]), s);
  }
  fprintf(stderr, "usage: lpgemm gate DIR M N K BATCH SEED GRID | bench REPS name:m:n:k[:batch]...\n");
  return 2;
}
#endif
