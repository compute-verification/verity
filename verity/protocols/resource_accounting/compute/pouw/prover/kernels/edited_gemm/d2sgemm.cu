// Dither-2S's linear (SKIP's arm adopted-bd-lm, Result 4) on sm_120a, on the
// honest Dither-1S kernel's mainloop (egemm.cu's e_l7qs: duo every 32 steps, absorbs 7 MMAs behind, consumers asleep on
// the stage barrier) with no tail stage.
//
// A linear Y = X W^T served as the last checked word of a product whose operands are both dithered with fresh bits (the
// int8 step's `dither` on both sides, benchmarks/ppl/steps.py): X = clamp(floor(x / s), -126, 126) + b and W' = W0 + c,
// W0 = clamp(floor(w / s_w), -126, 126) the cached weights, b and c fresh bits an entry a call. No U and no peel: every
// cell commits its running sums T_t = sum_{k' < 32 (t + 1)} X W'^T mod 2^32 (t < k / 32), each folded by the duo chains
// (derive.py's tile_receipts over T_1 .. T_last), and the served output is T_last = X W'^T, exact in s32.
// Noise from the per-call key (sha_pass.cuh's Sig; SHA-512(key || BE64(tag << 56 | j))): b as egemm.cu's (tag 7), c in
// the same patch layout over W0 (tag 8: block p = (row / 4) (k / 128) + col / 128, entry e = (row % 4) 128 + col % 128,
// bit e % 8 of digest byte e / 8), as cgemm.cu's c.
// Kernels:
//   d2quant          the dithered quantizer: egemm.cu's quant<true> at Dither-2S's range, without S's limbs
//   d2wedit          W' = W0 + c, a block 8 rows (two patch rows), its first 2 k / 128 threads hashing c's patches
//   d2qwe            d2quant and d2wedit in one launch (egemm.cu's qwe<2> block order)
//   d_*              the GEMM: e_l7qs's mainloop over X and W' with no tail stage (d_l7qs; d_l2q egemm.cu's e_l2q
//                    schedule, for the receipts' cross-check; d_plains no receipts)
// Modes:
//   d2sgemm gate DIR M N K SEED     poisoned buffers: each d_* kernel's output against cuBLASLt's X W'^T, d_l2q's receipts
//                                   against d_l7qs's, d2qwe's operands against d2quant's and d2wedit's, the overlap arm's
//                                   three calls, a no-write control; three tiles' x, W0, X, W', Y and receipts to DIR for
//                                   d2sgemm_ref.py
//   d2sgemm bench REPS M N K        hot items interleaved across D2_ARMS (egemm.cu's arms too), one JSON line
//   d2sgemm one ARM M N K           three warm-up calls, then one between cudaProfilerStart and Stop (for ncu)
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a d2sgemm.cu -lcuda -lcublasLt -lnvidia-ml
#define EGEMM_LIB
#include "egemm.cu"

constexpr uint32_t C_TAG = 8;
constexpr int D2_TOP = 126;   // |floor(x / s)| and |W0| <= 126, so X and W' are in [-126, 127]

// four bits to four int8 bytes 0/1
DEV uint32_t bits01(uint32_t nib) { return (nib * 0x00204081u) & 0x01010101u; }

// x (rows x k bf16) to X = clamp(floor(x inv_s), -126, 126) + b: egemm.cu's quant_blk<true>'s layout (a warp 4 rows, its
// lanes 8 columns each, chunks of 4096 columns whose 32 patches its lanes hash), no S. dgs: 8 x 512 words of shared memory
DEV void d2quant_blk(int blk, uint32_t* dgs, const uint16_t* __restrict__ x, int8_t* __restrict__ X, int rows, int k, float inv_s,
                     const Sig& sg) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, grp = blk * 8 + warp;
  if (grp * 4 >= rows) return;
  const int kb = k / 128;
  uint32_t* dg = dgs + 512 * warp;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dg);
#pragma unroll 1
  for (int p0 = 0; p0 < kb; p0 += 32) {
    const int np = min(32, kb - p0), steps = np / 2;
    if (p0) __syncwarp();
    if (lane < np) {
      uint64_t st[8];
      sha_block(sg.w, (uint64_t)DITHER_TAG << 56 | ((uint64_t)grp * kb + p0 + lane), st);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        dg[16 * lane + 2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
        dg[16 * lane + 2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
      }
    }
    __syncwarp();
#pragma unroll 1
    for (int rr = 0; rr < 4; rr++) {
      const size_t r = (size_t)grp * 4 + rr;
      const uint4* src = reinterpret_cast<const uint4*>(x + r * k) + 16 * p0;
      uint2* dst = reinterpret_cast<uint2*>(X + r * k) + 16 * p0;
#pragma unroll 4
      for (int it = 0; it < steps; it++) {
        const int ch = it * 32 + lane;
        const uint4 w = __ldg(src + ch);
        const uint32_t bits = bytes[64 * (ch / 16) + 16 * rr + ch % 16];
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        int qv[8];
#pragma unroll
        for (int h = 0; h < 4; h++) {
          const float f0 = __uint_as_float(ww[h] << 16) * inv_s, f1 = __uint_as_float(ww[h] & 0xFFFF0000u) * inv_s;
          qv[2 * h] = min(max(__float2int_rd(f0), -D2_TOP), D2_TOP) + (int)((bits >> (2 * h)) & 1u);
          qv[2 * h + 1] = min(max(__float2int_rd(f1), -D2_TOP), D2_TOP) + (int)((bits >> (2 * h + 1)) & 1u);
        }
        const uint32_t lo = __byte_perm(__byte_perm(qv[0], qv[1], 0x0040), __byte_perm(qv[2], qv[3], 0x0040), 0x5410);
        const uint32_t hi = __byte_perm(__byte_perm(qv[4], qv[5], 0x0040), __byte_perm(qv[6], qv[7], 0x0040), 0x5410);
        dst[ch] = make_uint2(lo, hi);
      }
    }
  }
}
__global__ void __launch_bounds__(256) d2quant(const uint16_t* __restrict__ x, int8_t* __restrict__ X, int rows, int k, float inv_s,
                                               const __grid_constant__ Sig sg) {
  __shared__ uint32_t dg[8 * 512];
  d2quant_blk(blockIdx.x, dg, x, X, rows, k, inv_s, sg);
}

// W' = W0 + c: a block 8 rows (two patch rows of 4), its threads loading W0's first 16-byte chunks (8 a thread) while
// threads 0 .. 2 k / 128 - 1 hash c's patches. cs: k bytes of c's digests, patch j of the block at 64 j
constexpr int D2W_MAXK = 16128;
DEV void d2wedit_blk(int blk, uint8_t* cs, const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp, int k, const Sig& sg) {
  const int r0 = blk * 8, t = threadIdx.x, cpr = k / 16, total = 8 * cpr, kb = k / 128;
  constexpr int MAXC = 8;
  uint4 w[MAXC];
  const uint4* src = reinterpret_cast<const uint4*>(W + (size_t)r0 * k);
#pragma unroll
  for (int i = 0; i < MAXC; i++)
    if (t + 256 * i < total) w[i] = __ldg(src + t + 256 * i);
  for (int j = t; j < 2 * kb; j += blockDim.x) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)C_TAG << 56 | (uint64_t)((r0 / 4 + j / kb) * kb + j % kb), st);
    uint32_t* o = reinterpret_cast<uint32_t*>(cs + 64 * j);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      o[2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      o[2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
  }
  __syncthreads();
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
        const uint4 x = w[i];
        dst[c] = make_uint4(add_noise<false>(x.x, bits01(b16 & 0xF)), add_noise<false>(x.y, bits01((b16 >> 4) & 0xF)),
                            add_noise<false>(x.z, bits01((b16 >> 8) & 0xF)), add_noise<false>(x.w, bits01(b16 >> 12)));
      }
    }
  }
}
__global__ void __launch_bounds__(256) d2wedit(const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp, int k,
                                               const __grid_constant__ Sig sg) {
  __shared__ __align__(16) uint8_t cs[D2W_MAXK];
  d2wedit_blk(blockIdx.x, cs, W, Wp, k, sg);
}
// d2quant and d2wedit in one launch under one key: every third block quantizes while quantizer blocks remain (egemm.cu's
// qwe<2>), the rest edit 8 rows of W0 each; 3 blocks an SM
__global__ void __launch_bounds__(256, 3) d2qwe(const uint16_t* __restrict__ x, int8_t* __restrict__ X, int rows, int k, float inv_s,
                                                int qb, const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp,
                                                const __grid_constant__ Sig sg) {
  __shared__ __align__(16) uint32_t sm[D2W_MAXK / 4 > 8 * 512 ? D2W_MAXK / 4 : 8 * 512];
  const int b = blockIdx.x, nw = (int)gridDim.x - qb;
  const int Q = min(qb, ((int)gridDim.x + 2) / 3);
  int q = b % 3 == 0 && b / 3 < qb ? b / 3 : -1;
  const int e = b - min((b + 2) / 3, qb);
  if (q < 0 && e >= nw) q = Q + (e - nw);
  if (q >= 0) d2quant_blk(q, sm, x, X, rows, k, inv_s, sg);
  else d2wedit_blk(e, reinterpret_cast<uint8_t*>(sm), W, Wp, k, sg);
}

// W0 in [-126, 126]; the gate's rows 0 and 1 at 126 and -126
__global__ void w0fill(int8_t* W, size_t count, uint32_t seed, size_t k, int ends) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)i) ^ mix32((uint32_t)(i >> 32) + 0x66u));
    int v = (int)(h % (2u * D2_TOP + 1u)) - D2_TOP;
    if (ends && i < k) v = D2_TOP;
    else if (ends && i < 2 * k) v = -D2_TOP;
    W[i] = (int8_t)v;
  }
}

// ---------------------------------------------------------------- the GEMM
// e_body without the peel: the mainloop's stages are the k / 128 of X and W' (no tail stage), T_1 .. T_last absorbed by
// tile_il's schedule and flush, and the served output is the accumulator T_last. BAT (Q K^T): A's rows are the problems
// stacked, 2^a.batch row tiles each, every one against its own N rows of B. !STORE: the served output is consumed in
// registers as a fused attention's softmax would start, its row maxima atomically into a.U (one int a row)
template <int RK, int LAG, int TIE, bool SLEEP, bool BAT = false, bool STORE = true> DEV void d_body(const Args& a) {
  using L = EL<RK, LAG, TIE, SLEEP, 0, 0, BAT>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps, a.M, a.N, a.K, a.group, false, 1);
  if constexpr (BAT) ml.bsh = a.batch;
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
    const int row0 = tm * 128 + wm * 32 + g, col0 = tn * 128 + wn * 32 + 2 * q;
    if constexpr (STORE) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          int* p = a.U + (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni;
          *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
          *reinterpret_cast<int2*>(p + (size_t)8 * a.N) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    } else {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          int v = max(acc[mi][0][2 * h], acc[mi][0][2 * h + 1]);
#pragma unroll
          for (int ni = 1; ni < NI; ni++) v = max(v, max(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]));
          v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
          v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
          if (q == 0) atomicMax(a.U + row0 + 16 * mi + 8 * h, v);
        }
    }
    if constexpr (L::R) {
      uint4* r = reinterpret_cast<uint4*>(a.R) + ((size_t)tile_id * L::NCT + ctid) * 2;
      r[0] = make_uint4(finish<DUO>(ch, 0), finish<DUO>(ch, 1), finish<DUO>(ch, 2), finish<DUO>(ch, 3));
      r[1] = make_uint4(finish<DUO>(ch, 4), finish<DUO>(ch, 5), finish<DUO>(ch, 6), finish<DUO>(ch, 7));
    }
  }
}
#define D2GEMM(NAME, RK, LAG, TIE, SLEEP) \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ Args a) { d_body<RK, LAG, TIE, SLEEP>(a); }
D2GEMM(d_l7qs, DUO, 7, 4, true)
D2GEMM(d_l2q, DUO, 2, 4, false)
D2GEMM(d_plains, NONE, 0, 0, true)
#define D2QK(NAME, RK, LAG, TIE, SLEEP, STORE) \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ Args a) { d_body<RK, LAG, TIE, SLEEP, true, STORE>(a); }
D2QK(qk_l7qs, DUO, 7, 4, true, true)
D2QK(qk_l2q, DUO, 2, 4, false, true)
D2QK(qk_plains, NONE, 0, 0, true, true)
D2QK(qk_l7qs_m, DUO, 7, 4, true, false)
D2QK(qk_plains_m, NONE, 0, 0, true, false)

// ---------------------------------------------------------------- Q K^T's operands
// Q K^T with both operands dithered (benchmarks/ppl/steps.py's int8_attention at q = k = "dither"): Q' = clamp(floor(q
// inv_s), -126, 126) + b and K'_h = clamp(floor(k_(h / rep) inv_s), -126, 126) + c_h, fresh bits an entry, K's for every
// query head; one inv_s for the tensor (steps.py's scale is per (head, token): a scale only multiplies before the floor).
// Both written head-major (row h T + t, 128 bytes) from q [T][H][128] and k [T][Hkv][128] bf16, the bits in the patch
// layout over those rows (k = 128: block row / 4; b tag 7, c tag 8, under the call's key). !DITHER: the plain quantizer,
// round(x inv_s) clamped to [-127, 127]. A warp 128 rows, its lanes hashing their 32 patches, then 4 columns a lane a row
template <bool DITHER>
__global__ void __launch_bounds__(64) qkquant(const uint16_t* __restrict__ src, int8_t* __restrict__ dst, int lgT, int hs,
                                              int lgrep, float inv_s, uint32_t tag, const __grid_constant__ Sig sg) {
  __shared__ uint32_t dgs[2 * 512];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, row0 = (blockIdx.x * 2 + warp) * 128;
  uint32_t* dg = dgs + 512 * warp;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dg);
  if constexpr (DITHER) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)tag << 56 | (uint64_t)(row0 / 4 + lane), st);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      dg[16 * lane + 2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      dg[16 * lane + 2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
    __syncwarp();
  }
  const int tmask = (1 << lgT) - 1;
#pragma unroll 4
  for (int rr = 0; rr < 128; rr++) {
    const int r = row0 + rr, t = r & tmask, h = r >> lgT;
    const uint2 w = __ldg(reinterpret_cast<const uint2*>(src + ((size_t)t * hs + (h >> lgrep)) * 128) + lane);
    const float f[4] = {__uint_as_float(w.x << 16) * inv_s, __uint_as_float(w.x & 0xFFFF0000u) * inv_s,
                        __uint_as_float(w.y << 16) * inv_s, __uint_as_float(w.y & 0xFFFF0000u) * inv_s};
    int v[4];
    if constexpr (DITHER) {
      const uint32_t bits = (uint32_t)bytes[16 * rr + lane / 2] >> (4 * (lane & 1));
#pragma unroll
      for (int i = 0; i < 4; i++) v[i] = min(max(__float2int_rd(f[i]), -D2_TOP), D2_TOP) + (int)((bits >> i) & 1u);
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) v[i] = min(max(__float2int_rn(f[i]), -127), 127);
    }
    reinterpret_cast<uint32_t*>(dst + (size_t)r * 128)[lane] =
        __byte_perm(__byte_perm(v[0], v[1], 0x0040), __byte_perm(v[2], v[3], 0x0040), 0x5410);
  }
}
// S's row maxima, the reference for the !STORE kernels
__global__ void rowmax_ref(const int* __restrict__ S, int* __restrict__ mx, size_t rows, int n) {
  for (size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += (size_t)gridDim.x * blockDim.x) {
    int v = S[r * n];
    for (int j = 1; j < n; j++) v = max(v, S[r * n + j]);
    mx[r] = v;
  }
}

// ---------------------------------------------------------------- host
struct DKn { const char* name; Kern k; const char* kind; };
static const DKn DKERN[] = {{"d_l7qs", d_l7qs, "duo32"}, {"d_l2q", d_l2q, "duo32"}, {"d_plains", d_plains, ""}};
static const DKn* dkern(const std::string& n) {
  for (auto& e : DKERN) if (n == e.name) return &e;
  return nullptr;
}
// egemm.cu's EP (its operands, cuBLASLt plans and arms) and Dither-2S's own: W0, X from d2quant, W' from d2wedit (and the
// overlap arm's two W' buffers), the fused producer's Xf and Wf, cuBLASLt's X W'^T plans as the reference
struct DP {
  EP E;
  int8_t *W0, *X, *Wp, *Wo[2], *Xf, *Wf;
  int *Y, *Yr, *Yr2;
  uint32_t *R, *R2;
  Maps *mD, *mO[2];
  LtG ref, refo[2];
  uint32_t inst = 0, ov = 0;
};
static void dq(DP& D, cudaStream_t st, const Sig& s) {
  EP& P = D.E;
  d2quant<<<(P.m + QROWS - 1) / QROWS, 256, 0, st>>>(P.x, D.X, P.m, P.k, P.inv_s, s);
}
static void dw(DP& D, cudaStream_t st, int8_t* Wp, const Sig& s) {
  EP& P = D.E;
  d2wedit<<<P.n / 8, 256, 0, st>>>(reinterpret_cast<const uint8_t*>(D.W0), reinterpret_cast<uint8_t*>(Wp), P.k, s);
}
static void dqwe(DP& D, int8_t* X, int8_t* Wp, const Sig& s) {
  EP& P = D.E;
  const int qb = (P.m + QROWS - 1) / QROWS;
  d2qwe<<<qb + P.n / 8, 256, 0, P.st>>>(P.x, X, P.m, P.k, P.inv_s, qb, reinterpret_cast<const uint8_t*>(D.W0),
                                         reinterpret_cast<uint8_t*>(Wp), s);
}
static void drun(DP& D, Kern kern, int* Y, uint32_t* R, const Maps* maps) {
  EP& P = D.E;
  Args a{maps, Y, R, P.m, P.n, P.k, 8, P.seed, 0, 1, 0, 0, {0, 0, 0, 0}, nullptr, 0};
  kern<<<P.grid, MLW::NT, CSMEM, P.st>>>(a);
}
// call i of the overlap arm: W'_(i+1) under key i + 1 built on st2 into buffer (i + 1) % 2 while st runs call i (d2quant
// under key i, then the GEMM on buffer i % 2, built under key i by call i - 1 or the setup); st then waits for st2
static void dov_call(DP& D, Kern k) {
  EP& P = D.E;
  const int b = D.ov & 1;
  CK(cudaEventRecord(P.ef, P.st));
  CK(cudaStreamWaitEvent(P.st2, P.ef));
  dw(D, P.st2, D.Wo[b ^ 1], osig(P, D.ov + 1));
  dq(D, P.st, osig(P, D.ov));
  drun(D, k, D.Y, D.R, D.mO[b]);
  CK(cudaEventRecord(P.ej, P.st2));
  CK(cudaStreamWaitEvent(P.st, P.ej));
  D.ov++;
}
static void dsetup(DP& D, int m, int n, int k, uint32_t seed, bool gate) {
  EP& P = D.E;
  esetup(P, m, n, k, seed, gate);
  const size_t xb = (size_t)m * k, wb = (size_t)n * k;
  CK(cudaMalloc(&D.W0, wb)); CK(cudaMalloc(&D.X, xb)); CK(cudaMalloc(&D.Wp, wb));
  CK(cudaMalloc(&D.Wo[0], wb)); CK(cudaMalloc(&D.Wo[1], wb)); CK(cudaMalloc(&D.Xf, xb)); CK(cudaMalloc(&D.Wf, wb));
  CK(cudaMalloc(&D.Y, P.ny * 4)); CK(cudaMalloc(&D.Yr, P.ny * 4)); CK(cudaMalloc(&D.Yr2, P.ny * 4));
  CK(cudaMalloc(&D.R, P.nr * 4)); CK(cudaMalloc(&D.R2, P.nr * 4));
  w0fill<<<4096, 256, 0, P.st>>>(D.W0, wb, mix32(seed ^ 0x5555u), k, gate);
  const Sig s0 = sigma(seed, 0xFFFFFFFFu);
  dq(D, P.st, s0);
  dw(D, P.st, D.Wp, s0);
  dw(D, P.st, D.Wo[0], osig(P, 0));
  dqwe(D, D.Xf, D.Wf, s0);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  D.ref = ltg(I8, m, n, k, k, k, n, D.X, D.Wp, D.Yr, 0, P.st);
  for (int b = 0; b < 2; b++) D.refo[b] = ltg(I8, m, n, k, k, k, n, D.X, D.Wo[b], D.Yr2, 0, P.st);
  // no tail stage reads pa and pb; the prologue only prefetches their maps
  D.mD = upload_maps(Maps{tmap(D.X, k, m, 128), tmap(D.Wp, k, n, 128), tmap(D.X, k, m, 128), tmap(D.Wp, k, n, 128)});
  for (int b = 0; b < 2; b++)
    D.mO[b] = upload_maps(Maps{tmap(D.X, k, m, 128), tmap(D.Wo[b], k, n, 128), tmap(D.X, k, m, 128), tmap(D.Wo[b], k, n, 128)});
  for (auto& e : DKERN) CK(cudaFuncSetAttribute(e.k, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
}
static void dfree(DP& D) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)D.W0, (void*)D.X, (void*)D.Wp, (void*)D.Wo[0], (void*)D.Wo[1], (void*)D.Xf, (void*)D.Wf, (void*)D.Y,
                  (void*)D.Yr, (void*)D.Yr2, (void*)D.R, (void*)D.R2, (void*)D.mD, (void*)D.mO[0], (void*)D.mO[1]})
    CK(cudaFree(p));
  efree(D.E);
}
// an arm: a d_* kernel (on d2quant's and d2wedit's operands), d2quant, d2wedit, d2ops (both under one key), d2qwe (the
// fused producer), lt_d2s (cuBLASLt on Dither-2S's X and W': the plain GEMM on the same operands), dedited:KERN (d2quant,
// d2wedit, KERN under one key), deditedf:KERN (d2qwe, then KERN), dov:KERN (the overlap arm), or any of egemm.cu's arms
// (plain, plainq, e_plains, e_l7qs, editedq:e_l7qs, overlapq:e_l7qs, ...) on its own operands
static std::function<void(DP&)> darm(const std::string& a) {
  if (a == "d2quant") return [](DP& D) { dq(D, D.E.st, sigma(D.E.seed, D.inst++)); };
  if (a == "d2wedit") return [](DP& D) { dw(D, D.E.st, D.Wp, sigma(D.E.seed, D.inst++)); };
  if (a == "d2ops") return [](DP& D) { const Sig s = sigma(D.E.seed, D.inst++); dq(D, D.E.st, s); dw(D, D.E.st, D.Wp, s); };
  if (a == "d2qwe") return [](DP& D) { dqwe(D, D.X, D.Wp, sigma(D.E.seed, D.inst++)); };
  if (a == "lt_d2s") return [](DP& D) { ltg_run(D.ref, D.E.st); };
  auto kern_of = [&a](size_t at) {
    const DKn* k = dkern(a.substr(at));
    if (!k) { fprintf(stderr, "no kernel %s\n", a.c_str()); exit(2); }
    return k;
  };
  if (a.rfind("dedited:", 0) == 0) {
    const DKn* k = kern_of(8);
    return [k](DP& D) {
      const Sig s = sigma(D.E.seed, D.inst++);
      dq(D, D.E.st, s); dw(D, D.E.st, D.Wp, s); drun(D, k->k, D.Y, D.R, D.mD);
    };
  }
  if (a.rfind("deditedf:", 0) == 0) {
    const DKn* k = kern_of(9);
    return [k](DP& D) { dqwe(D, D.X, D.Wp, sigma(D.E.seed, D.inst++)); drun(D, k->k, D.Y, D.R, D.mD); };
  }
  if (a.rfind("dov:", 0) == 0) {
    const DKn* k = kern_of(4);
    return [k](DP& D) { dov_call(D, k->k); };
  }
  if (const DKn* k = dkern(a)) return [k](DP& D) { drun(D, k->k, D.Y, D.R, D.mD); };
  auto f = earm(a);
  return [f](DP& D) { f(D.E); };
}

static int dgate(const std::string& dir, int m, int n, int k, uint32_t seed) {
  DP D;
  dsetup(D, m, n, k, seed, true);
  EP& P = D.E;
  const size_t xb = (size_t)m * k, wb = (size_t)n * k;
  for (auto [p, b] : {std::pair<void*, size_t>{D.X, xb}, {D.Wp, wb}, {D.Xf, xb}, {D.Wf, wb}, {D.Y, P.ny * 4}, {D.Yr, P.ny * 4},
                      {D.Yr2, P.ny * 4}, {D.R, P.nr * 4}, {D.R2, P.nr * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, P.st));
  const Sig s = sigma(seed, 0);
  dq(D, P.st, s);
  dw(D, P.st, D.Wp, s);
  ltg_run(D.ref, P.st);
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
      if (Y) {
        std::vector<int> yt(128 * 128);
        for (int r = 0; r < 128; r++)
          CK(cudaMemcpy(yt.data() + r * 128, Y + (size_t)(tm * 128 + r) * n + tn * 128, 128 * 4, cudaMemcpyDeviceToHost));
        dump(dir, ("Y_" + name + sfx).c_str(), yt.data(), yt.size() * 4, false);
      }
    }
  };
  // d_l7qs: its output against cuBLASLt's X W'^T; its receipts are the other schedule's reference
  CK(cudaMemsetAsync(D.Y, 0xA5, P.ny * 4, P.st));
  drun(D, d_l7qs, D.Y, D.R2, D.mD);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  {
    const unsigned long long y = ecmp(P, D.Y, D.Yr, P.ny);
    rec("d_l7qs", y, -1);
    ok &= y == 0;
    dump_tiles("d_l7qs", D.Y, D.R2);
  }
  for (const char* name : {"d_l2q", "d_plains"}) {
    const DKn* e = dkern(name);
    CK(cudaMemsetAsync(D.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(D.R, 0xA5, P.nr * 4, P.st));
    drun(D, e->k, D.Y, D.R, D.mD);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    const unsigned long long y = ecmp(P, D.Y, D.Yr, P.ny), r = *e->kind ? ecmp(P, D.R, D.R2, P.nr) : 0;
    rec(name, y, *e->kind ? (long long)r : -1);
    ok &= y == 0 && r == 0;
    if (*e->kind) dump_tiles(name, nullptr, D.R);
  }
  // the fused producer under the same key: its X and W' bitwise d2quant's and d2wedit's
  dqwe(D, D.Xf, D.Wf, s);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  {
    const unsigned long long fx = ecmp(P, D.Xf, D.X, xb / 4) + ecmp(P, D.Wf, D.Wp, wb / 4);
    rec("d2qwe_operands", fx, -1);
    ok &= fx == 0;
  }
  // the no-write control: poisoned, nothing run; both compares must fail
  CK(cudaMemsetAsync(D.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(D.R, 0xA5, P.nr * 4, P.st));
  const unsigned long long cy = ecmp(P, D.Y, D.Yr, P.ny), cr = ecmp(P, D.R, D.R2, P.nr);
  const bool control_rejected = cy > 0 && cr > 0;
  rec("control", cy, (long long)cr);
  std::string tj;
  for (int i = 0; i < 3; i++) {
    const int tm = tl[i][0], tn = tl[i][1];
    const std::string sfx = "_" + std::to_string(tm) + "_" + std::to_string(tn) + ".bin";
    dump(dir, ("x" + sfx).c_str(), P.x + (size_t)tm * 128 * k, 128 * (size_t)k * 2, true);
    dump(dir, ("X" + sfx).c_str(), D.X + (size_t)tm * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("W" + sfx).c_str(), D.W0 + (size_t)tn * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("Wp" + sfx).c_str(), D.Wp + (size_t)tn * 128 * k, 128 * (size_t)k, true);
    tj += std::string(i ? ", " : "") + "[" + std::to_string(tm) + ", " + std::to_string(tn) + "]";
  }
  // the overlap arm: three calls (both buffers, keys osig 0-2), each output against cuBLASLt's on that call's X and
  // W' buffer and its receipts against d_l2q's on the same operands
  for (int c = 0; c < 3; c++) {
    const int b = D.ov & 1;
    CK(cudaMemsetAsync(D.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(D.R, 0xA5, P.nr * 4, P.st));
    CK(cudaMemsetAsync(D.R2, 0xA5, P.nr * 4, P.st)); CK(cudaMemsetAsync(D.Yr2, 0xA5, P.ny * 4, P.st));
    dov_call(D, d_l7qs);
    ltg_run(D.refo[b], P.st);
    drun(D, d_l2q, D.Yr, D.R2, D.mO[b]);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    const unsigned long long y = ecmp(P, D.Y, D.Yr2, P.ny), y2 = ecmp(P, D.Yr, D.Yr2, P.ny), r = ecmp(P, D.R, D.R2, P.nr);
    rec("dov_d_l7qs_" + std::to_string(c), y + y2, (long long)r);
    ok &= y == 0 && y2 == 0 && r == 0;
  }
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"inv_s\": %.9g, \"grid\": %d, \"key\": [\"%016llx\", \"%016llx\", "
              "\"%016llx\", \"%016llx\"], \"tiles\": [%s], \"kinds\": {\"d_l7qs\": \"duo32\", \"d_l2q\": \"duo32\"}, \"device\": {%s}, "
              "\"device_ok\": %s, \"control_rejected\": %s}\n",
          m, n, k, seed, P.inv_s, P.grid, (unsigned long long)s.w[0], (unsigned long long)s.w[1], (unsigned long long)s.w[2],
          (unsigned long long)s.w[3], tj.c_str(), res.c_str(), ok ? "true" : "false", control_rejected ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"device_ok\": %s, \"control_rejected\": %s, \"device\": {%s}}\n",
         dir.c_str(), m, n, k, ok ? "true" : "false", control_rejected ? "true" : "false", res.c_str());
  dfree(D);
  return ok && control_rejected ? 0 : 1;
}

static int dbench(int reps, int m, int n, int k) {
  Nv nv;
  const std::vector<std::string> names =
      split(getenv("D2_ARMS"), "plain,plainq,lt_d2s,d_plains,e_l7qs,d_l7qs,editedq:e_l7qs,dedited:d_l7qs,dov:d_l7qs");
  DP D;
  dsetup(D, m, n, k, 0x5EED0000u + m + n + k, false);
  std::vector<std::function<void(DP&)>> arms;
  for (auto& a : names) arms.push_back(darm(a));
  const int NA = (int)arms.size();
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_arm = [&](int j, int calls) {
    CK(cudaEventRecord(e0, D.E.st));
    for (int c = 0; c < calls; c++) arms[j](D);
    CK(cudaEventRecord(e1, D.E.st));
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
  printf("{\"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d, \"lt_algo\": %d, \"lt_d2s_algo\": %d, \"arms\": {", m, n, k, D.E.grid,
         D.E.plain.algo_id, D.ref.algo_id);
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
  dfree(D);
  return 0;
}

static int done(const std::string& arm, int m, int n, int k) {
  DP D;
  dsetup(D, m, n, k, 0x5EED0000u + m + n + k, false);
  auto f = darm(arm);
  for (int i = 0; i < 3; i++) f(D);
  CK(cudaStreamSynchronize(D.E.st));
  CK(cudaProfilerStart());
  f(D);
  CK(cudaStreamSynchronize(D.E.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"one\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d}\n", arm.c_str(), m, n, k, D.E.grid);
  dfree(D);
  return 0;
}

// ---------------------------------------------------------------- Q K^T: host
// H query heads, Hkv KV heads, T tokens (one sequence's prefill, every key against every query: no causal skip). Q' and
// K' head-major (H T x 128 each, K' a copy a query head with its own bits), S = Q'_h K'_h^T stacked (H T x T s32)
struct QP {
  int H, Hkv, T, lgT, lgrep, bsh, grid;
  uint32_t seed, inst = 0;
  float inv_s = 1.f;
  uint16_t *q, *k;
  int8_t *Q, *K, *Qq, *Kq;
  int *S, *Sr, *mx, *mxr;
  uint32_t *R, *R2;
  size_t ns, nr, nq;
  Maps *mD, *mP;
  LtG lt, ltq;
  cudaStream_t st;
  unsigned long long* bad;
};
static int lg2(int x) {
  int l = 0;
  while ((1 << l) < x) l++;
  return (1 << l) == x ? l : -1;
}
static unsigned long long qcmp(QP& P, const void* a, const void* b, size_t words) {
  CK(cudaMemsetAsync(P.bad, 0, 8, P.st));
  cmp_words<<<1024, 256, 0, P.st>>>((const uint32_t*)a, (const uint32_t*)b, words, P.bad);
  unsigned long long h;
  CK(cudaMemcpyAsync(&h, P.bad, 8, cudaMemcpyDeviceToHost, P.st));
  CK(cudaStreamSynchronize(P.st));
  return h;
}
static void qq(QP& P, bool dither, const Sig& s) {
  const int blocks = P.H * P.T / 256;
  if (dither) {
    qkquant<true><<<blocks, 64, 0, P.st>>>(P.q, P.Q, P.lgT, P.H, 0, P.inv_s, DITHER_TAG, s);
    qkquant<true><<<blocks, 64, 0, P.st>>>(P.k, P.K, P.lgT, P.Hkv, P.lgrep, P.inv_s, C_TAG, s);
  } else {
    qkquant<false><<<blocks, 64, 0, P.st>>>(P.q, P.Qq, P.lgT, P.H, 0, P.inv_s, 0, s);
    qkquant<false><<<blocks, 64, 0, P.st>>>(P.k, P.Kq, P.lgT, P.Hkv, P.lgrep, P.inv_s, 0, s);
  }
}
static void qrun(QP& P, Kern kern, int* U, uint32_t* R, const Maps* maps) {
  Args a{maps, U, R, P.H * P.T, P.T, 128, 8, P.seed, 0, P.bsh, 0, 0, {0, 0, 0, 0}, nullptr, 0};
  kern<<<P.grid, MLW::NT, CSMEM, P.st>>>(a);
}
static void qsetup(QP& P, int H, int Hkv, int T, uint32_t seed, bool gate) {
  P.H = H; P.Hkv = Hkv; P.T = T; P.seed = seed;
  P.lgT = lg2(T); P.lgrep = Hkv > 0 && H % Hkv == 0 ? lg2(H / Hkv) : -1; P.bsh = lg2(T / 128);
  if (T % 1024 || P.lgT < 0 || P.lgrep < 0 || (H * T) % 256) {
    fprintf(stderr, "qk %d %d %d: T a power of 2 and a multiple of 1024 (EL's raster groups inside a head), H / Hkv a power of 2\n", H, Hkv, T);
    exit(2);
  }
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const int tiles = (H * T / 128) * (T / 128);
  P.grid = std::min(tiles, nsm);
  P.nq = (size_t)H * T * 128;
  P.ns = (size_t)H * T * T;
  P.nr = (size_t)tiles * 512 * 8;
  CK(cudaStreamCreate(&P.st));
  CK(cudaMalloc(&P.q, (size_t)T * H * 128 * 2)); CK(cudaMalloc(&P.k, (size_t)T * Hkv * 128 * 2));
  CK(cudaMalloc(&P.Q, P.nq)); CK(cudaMalloc(&P.K, P.nq)); CK(cudaMalloc(&P.Qq, P.nq)); CK(cudaMalloc(&P.Kq, P.nq));
  CK(cudaMalloc(&P.S, P.ns * 4)); CK(cudaMalloc(&P.Sr, P.ns * 4));
  CK(cudaMalloc(&P.mx, (size_t)H * T * 4)); CK(cudaMalloc(&P.mxr, (size_t)H * T * 4));
  CK(cudaMalloc(&P.R, P.nr * 4)); CK(cudaMalloc(&P.R2, P.nr * 4));
  CK(cudaMalloc(&P.bad, 8));
  xfill<<<4096, 256, 0, P.st>>>(P.q, (size_t)T * H * 128, mix32(seed ^ 0x3333u), 128, gate);
  xfill<<<4096, 256, 0, P.st>>>(P.k, (size_t)T * Hkv * 128, mix32(seed ^ 0x6666u), 128, gate);
  qq(P, true, sigma(seed, 0xFFFFFFFFu));
  qq(P, false, sigma(seed, 0xFFFFFFFFu));
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  P.lt = ltg(I8, T, T, 128, 128, 128, T, P.Q, P.K, P.Sr, 0, P.st, H);
  P.ltq = ltg(I8, T, T, 128, 128, 128, T, P.Qq, P.Kq, P.Sr, 0, P.st, H);
  P.mD = upload_maps(Maps{tmap(P.Q, 128, H * T, 128), tmap(P.K, 128, H * T, 128), tmap(P.Q, 128, H * T, 128), tmap(P.K, 128, H * T, 128)});
  P.mP = upload_maps(Maps{tmap(P.Qq, 128, H * T, 128), tmap(P.Kq, 128, H * T, 128), tmap(P.Qq, 128, H * T, 128), tmap(P.Kq, 128, H * T, 128)});
  for (Kern kern : {qk_l7qs, qk_l2q, qk_plains, qk_l7qs_m, qk_plains_m})
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
}
static void qfree(QP& P) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)P.q, (void*)P.k, (void*)P.Q, (void*)P.K, (void*)P.Qq, (void*)P.Kq, (void*)P.S, (void*)P.Sr, (void*)P.mx,
                  (void*)P.mxr, (void*)P.R, (void*)P.R2, (void*)P.bad, (void*)P.mD, (void*)P.mP})
    CK(cudaFree(p));
  CK(cudaStreamDestroy(P.st));
}
// Q K^T's arms: qk_l7qs, qk_l2q (S written; on Q', K'), qk_plains (the plain mainloop on the rounded operands), the _m
// twins (S's row maxima only), qk_lt / qk_ltq (cuBLASLt's batched GEMM on Q', K' / on the rounded operands), qkq_d and
// qkq_plain (both operands' quantizers, dithered under a fresh key / rounded), and the served calls: qk_served (qkq_d,
// qk_l7qs under one key), qk_plainq (qkq_plain, qk_ltq), qk_served_m (qkq_d, qk_l7qs_m), qk_plainq_m (qkq_plain, qk_plains_m)
static std::function<void(QP&)> qarm(const std::string& a) {
  auto mxr = [](QP& P) { CK(cudaMemsetAsync(P.mx, 0x80, (size_t)P.H * P.T * 4, P.st)); };
  if (a == "qk_l7qs") return [](QP& P) { qrun(P, qk_l7qs, P.S, P.R, P.mD); };
  if (a == "qk_l2q") return [](QP& P) { qrun(P, qk_l2q, P.S, P.R, P.mD); };
  if (a == "qk_plains") return [](QP& P) { qrun(P, qk_plains, P.S, nullptr, P.mP); };
  if (a == "qk_l7qs_m") return [mxr](QP& P) { mxr(P); qrun(P, qk_l7qs_m, P.mx, P.R, P.mD); };
  if (a == "qk_plains_m") return [mxr](QP& P) { mxr(P); qrun(P, qk_plains_m, P.mx, nullptr, P.mP); };
  if (a == "qk_lt") return [](QP& P) { ltg_run(P.lt, P.st); };
  if (a == "qk_ltq") return [](QP& P) { ltg_run(P.ltq, P.st); };
  if (a == "qkq_d") return [](QP& P) { qq(P, true, sigma(P.seed, P.inst++)); };
  if (a == "qkq_plain") return [](QP& P) { qq(P, false, sigma(P.seed, 0)); };
  if (a == "qk_served") return [](QP& P) { qq(P, true, sigma(P.seed, P.inst++)); qrun(P, qk_l7qs, P.S, P.R, P.mD); };
  if (a == "qk_plainq") return [](QP& P) { qq(P, false, sigma(P.seed, 0)); ltg_run(P.ltq, P.st); };
  if (a == "qk_served_m") return [mxr](QP& P) { qq(P, true, sigma(P.seed, P.inst++)); mxr(P); qrun(P, qk_l7qs_m, P.mx, P.R, P.mD); };
  if (a == "qk_plainq_m") return [mxr](QP& P) { qq(P, false, sigma(P.seed, 0)); mxr(P); qrun(P, qk_plains_m, P.mx, nullptr, P.mP); };
  fprintf(stderr, "no qk arm %s\n", a.c_str());
  exit(2);
}
static int qgate(const std::string& dir, int H, int Hkv, int T, uint32_t seed) {
  QP P;
  qsetup(P, H, Hkv, T, seed, true);
  for (auto [p, b] : {std::pair<void*, size_t>{P.Q, P.nq}, {P.K, P.nq}, {P.S, P.ns * 4}, {P.Sr, P.ns * 4}, {P.R, P.nr * 4},
                      {P.R2, P.nr * 4}, {P.mx, (size_t)H * T * 4}, {P.mxr, (size_t)H * T * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, P.st));
  const Sig s = sigma(seed, 0);
  qq(P, true, s);
  ltg_run(P.lt, P.st);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  rowmax_ref<<<(H * T + 255) / 256, 256, 0, P.st>>>(P.Sr, P.mxr, (size_t)H * T, T);
  bool ok = true;
  std::string res;
  auto rec = [&](const std::string& name, unsigned long long y, long long r) {
    res += (res.empty() ? "" : ", ") + std::string("\"") + name + "\": {\"y_bad\": " + std::to_string(y) + ", \"r_bad\": " + std::to_string(r) + "}";
    ok &= y == 0 && r <= 0;
  };
  const int tiles_m = H * T / 128, tiles_n = T / 128;
  const int tl[3][2] = {{0, 0}, {tiles_m / 2 + tiles_n / 2, tiles_n / 2 - 1}, {tiles_m - 1, tiles_n - 1}};
  auto dump_tiles = [&](const std::string& name, const int* Y, const uint32_t* R) {
    for (int i = 0; i < 3; i++) {
      const int tm = tl[i][0], tn = tl[i][1];
      const std::string sfx = "_" + std::to_string(tm) + "_" + std::to_string(tn) + ".bin";
      if (R) dump(dir, ("R_" + name + sfx).c_str(), R + (size_t)(tm * tiles_n + tn) * 512 * 8, 512 * 8 * 4, true);
      if (Y) {
        std::vector<int> yt(128 * 128);
        CK(cudaMemcpy2D(yt.data(), 128 * 4, Y + (size_t)tm * 128 * T + tn * 128, (size_t)T * 4, 128 * 4, 128, cudaMemcpyDeviceToHost));
        dump(dir, ("Y_" + name + sfx).c_str(), yt.data(), yt.size() * 4, false);
      }
    }
  };
  qrun(P, qk_l7qs, P.S, P.R2, P.mD);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  rec("qk_l7qs", qcmp(P, P.S, P.Sr, P.ns), -1);
  dump_tiles("qk_l7qs", P.S, P.R2);
  for (auto [name, kern, rk] : {std::tuple<const char*, Kern, bool>{"qk_l2q", qk_l2q, true}, {"qk_plains", qk_plains, false}}) {
    CK(cudaMemsetAsync(P.S, 0xA5, P.ns * 4, P.st)); CK(cudaMemsetAsync(P.R, 0xA5, P.nr * 4, P.st));
    qrun(P, kern, P.S, P.R, P.mD);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    rec(name, qcmp(P, P.S, P.Sr, P.ns), rk ? (long long)qcmp(P, P.R, P.R2, P.nr) : -1);
    if (rk) dump_tiles(name, nullptr, P.R);
  }
  for (auto [name, kern, rk] : {std::tuple<const char*, Kern, bool>{"qk_l7qs_m", qk_l7qs_m, true}, {"qk_plains_m", qk_plains_m, false}}) {
    CK(cudaMemsetAsync(P.mx, 0x80, (size_t)H * T * 4, P.st)); CK(cudaMemsetAsync(P.R, 0xA5, P.nr * 4, P.st));
    qrun(P, kern, P.mx, P.R, P.mD);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    rec(name, qcmp(P, P.mx, P.mxr, (size_t)H * T), rk ? (long long)qcmp(P, P.R, P.R2, P.nr) : -1);
  }
  // the plain quantizer's operands through the plain mainloop against cuBLASLt's on the same
  CK(cudaMemsetAsync(P.S, 0xA5, P.ns * 4, P.st)); CK(cudaMemsetAsync(P.Sr, 0x5A, P.ns * 4, P.st));
  ltg_run(P.ltq, P.st);
  qrun(P, qk_plains, P.S, nullptr, P.mP);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  rec("qk_plains_rounded", qcmp(P, P.S, P.Sr, P.ns), -1);
  CK(cudaMemsetAsync(P.S, 0xA5, P.ns * 4, P.st)); CK(cudaMemsetAsync(P.R, 0xA5, P.nr * 4, P.st));
  const unsigned long long cy = qcmp(P, P.S, P.Sr, P.ns), cr = qcmp(P, P.R, P.R2, P.nr);
  const bool control_rejected = cy > 0 && cr > 0;
  res += ", \"control\": {\"y_bad\": " + std::to_string(cy) + ", \"r_bad\": " + std::to_string(cr) + "}";
  std::string tj;
  for (int i = 0; i < 3; i++) {
    const int tm = tl[i][0], tn = tl[i][1], h = tm / tiles_n, t0 = (tm % tiles_n) * 128, k0 = tn * 128;
    const std::string sfx = "_" + std::to_string(tm) + "_" + std::to_string(tn) + ".bin";
    std::vector<uint16_t> xs(128 * 128), ks(128 * 128);
    CK(cudaMemcpy2D(xs.data(), 256, P.q + ((size_t)t0 * H + h) * 128, (size_t)H * 256, 256, 128, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy2D(ks.data(), 256, P.k + ((size_t)k0 * Hkv + (h >> P.lgrep)) * 128, (size_t)Hkv * 256, 256, 128, cudaMemcpyDeviceToHost));
    dump(dir, ("x" + sfx).c_str(), xs.data(), xs.size() * 2, false);
    dump(dir, ("xk" + sfx).c_str(), ks.data(), ks.size() * 2, false);
    dump(dir, ("X" + sfx).c_str(), P.Q + (size_t)tm * 128 * 128, 128 * 128, true);
    dump(dir, ("Wp" + sfx).c_str(), P.K + ((size_t)h * T + k0) * 128, 128 * 128, true);
    tj += std::string(i ? ", " : "") + "[" + std::to_string(tm) + ", " + std::to_string(tn) + "]";
  }
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"mode\": \"qk\", \"served\": \"qk_l7qs\", \"m\": %d, \"n\": %d, \"k\": 128, \"H\": %d, \"Hkv\": %d, \"T\": %d, "
              "\"seed\": %u, \"inv_s\": %.9g, \"grid\": %d, \"key\": [\"%016llx\", \"%016llx\", \"%016llx\", \"%016llx\"], "
              "\"tiles\": [%s], \"kinds\": {\"qk_l7qs\": \"duo32\", \"qk_l2q\": \"duo32\"}, \"device\": {%s}, \"device_ok\": %s, "
              "\"control_rejected\": %s}\n",
          H * T, T, H, Hkv, T, seed, P.inv_s, P.grid, (unsigned long long)s.w[0], (unsigned long long)s.w[1], (unsigned long long)s.w[2],
          (unsigned long long)s.w[3], tj.c_str(), res.c_str(), ok ? "true" : "false", control_rejected ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"qk\": [%d, %d, %d], \"device_ok\": %s, \"control_rejected\": %s, \"device\": {%s}}\n", dir.c_str(), H,
         Hkv, T, ok ? "true" : "false", control_rejected ? "true" : "false", res.c_str());
  qfree(P);
  return ok && control_rejected ? 0 : 1;
}
static int qbench(int reps, int H, int Hkv, int T) {
  Nv nv;
  const std::vector<std::string> names = split(getenv("QK_ARMS"),
      "qk_ltq,qk_lt,qk_plains,qk_l7qs,qk_l2q,qk_plains_m,qk_l7qs_m,qkq_plain,qkq_d,qk_plainq,qk_served,qk_plainq_m,qk_served_m");
  QP P;
  qsetup(P, H, Hkv, T, 0x5EED0000u + H + T, false);
  std::vector<std::function<void(QP&)>> arms;
  for (auto& a : names) arms.push_back(qarm(a));
  const int NA = (int)arms.size();
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_arm = [&](int j, int calls) {
    CK(cudaEventRecord(e0, P.st));
    for (int c = 0; c < calls; c++) arms[j](P);
    CK(cudaEventRecord(e1, P.st));
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
  printf("{\"qk\": [%d, %d, %d], \"m\": %d, \"n\": %d, \"k\": 128, \"grid\": %d, \"lt_algo\": %d, \"arms\": {", H, Hkv, T, H * T, T,
         P.grid, P.lt.algo_id);
  for (int j = 0; j < NA; j++) {
    printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", names[j].c_str(), median(ms[j]),
           median(mhz[j]), calls[j]);
    for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
    printf("]}");
  }
  printf("}}\n");
  fflush(stdout);
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  qfree(P);
  return 0;
}
static int qone(const std::string& arm, int H, int Hkv, int T) {
  QP P;
  qsetup(P, H, Hkv, T, 0x5EED0000u + H + T, false);
  auto f = qarm(arm);
  for (int i = 0; i < 3; i++) f(P);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaProfilerStart());
  f(P);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"one\": \"%s\", \"qk\": [%d, %d, %d], \"grid\": %d}\n", arm.c_str(), H, Hkv, T, P.grid);
  qfree(P);
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "qkgate")) return qgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 6 && !strcmp(argv[1], "qkbench")) return qbench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 6 && !strcmp(argv[1], "qkone")) return qone(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return dgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 6 && !strcmp(argv[1], "bench")) return dbench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 6 && !strcmp(argv[1], "one")) return done(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: d2sgemm gate DIR M N K SEED | bench REPS M N K | one ARM M N K | qkgate DIR H HKV T SEED | "
                  "qkbench REPS H HKV T | qkone ARM H HKV T\n");
  return 2;
}
