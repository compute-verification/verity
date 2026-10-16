// The adopted design's edited GEMM (decision 29) on sm_120a (RTX PRO 6000), as close to plain int8 as we can get it.
//
// A linear Y = X W^T (X m x k activations, W n x k weights capped at ±63) served as the adopted design's noised product:
// the served operand is the dithered X = clamp(floor(x / s), -127, 126) + b, b a fresh coin bit an entry, and the weights
// carry U in {±1..±64} (n x 32) peeled on the activations' side: W' = W + U Id^T (W'_jk = W_jk + U_{j, k mod 32}),
// C = X W'^T on the tensor cores, Y = C - S U^T with S = X Id (S_ic = sum_t X_{i, 32 t + c}, |S| <= 128 k / 32). The
// prover commits C's stride-32 running sums by duo receipts (fgemm.cu's; benchmarks/e2e_slice/derive.py's tile_receipts).
// Noise from a 256-bit per-call key (sha_pass.cuh's Sig, SHA-512(key || BE64(tag << 56 | j))): U as lpgemm.cu's LP_UW 64
// (tag 6, block j = rows 2j, 2j + 1, entry (row % 2) 32 + l digest byte, u = (v & 63) + 1, negated if v & 64); b in
// sha_noise.cuh's NS 1 patch layout under tag 7 (block p = (row / 4) (k / 128) + col / 128, entry e = (row % 4) 128 +
// col % 128, b = bit e % 8 of digest byte e / 8).
// Kernels:
//   quant<false>, quant<true>  the activation quantizer from bf16: plain (round) and dithered (floor + b, with S's limbs
//                    into the tail rows P_A = [S_hi | 0 | S_lo | 0], S = 256 S_hi + S_lo), so the dither is priced against
//                    the quantizer plain serving runs anyway (k <= 4096: a warp's 4 rows hash their 32 patches at once)
//   wedit            W' = W + U and the tail rows P_B = [U | 0 | U | 0], U's hashes in each block of 8 rows
//   e_*              the GEMM: lpgemm.cu's 16-warp 128 x 128 TMA mainloop (fgemm.cu's ML) with the receipts' absorbs
//                    interleaved with the MMAs, and the peel as a 2-atom tail (acc -= 256 S_hi U^T + S_lo U^T, the same
//                    rows t_duo32's 4-atom tail reads). Interleaving: the absorb of tile j's running sum runs LAG MMAs
//                    after tile j's MMA (each value is still in its accumulator then, so the absorb order and the receipts
//                    are t_duo32's), and with TIE each MMA's accumulator is xored with (z & zero), z the last chain the
//                    previous absorb wrote and zero a kernel argument that is 0, so ptxas can't hoist a run of MMAs
//                    ahead of the absorbs (without it, it issues IMMA NOP IMMA NOP ... and then the absorbs in a burst)
// Modes:
//   egemm gate DIR M N K SEED       every arm once on poisoned buffers: each GEMM's output against cuBLASLt's X W^T, each
//                                   receipt arm's receipts against t_duo32's, a no-write control; tiles to DIR for egemm_ref.py
//   egemm bench REPS M N K          hot items (about 20 ms each) interleaved across EG_ARMS, one JSON line
//   egemm il REPS ARM BASE M N K    per call, ARM and BASE alternating (the slice's timing), one JSON line
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a egemm.cu -lcuda -lcublasLt -lnvidia-ml
#define LP_LIB
#define LP_NO_ROUTES
#define LP_UW 64
#include "../masked_gemm/lpgemm.cu"

#include <functional>

constexpr uint32_t DITHER_TAG = 7;
using MLW = ML<128, 128, 32, 3>;

// d ^= z & zero in one LOP3 (LUT 0x78 = a ^ (b & c)); ptxas otherwise splits it into an AND and an XOR
DEV void tie(int& d, uint32_t z, uint32_t zero) { asm("lop3.b32 %0, %0, %1, %2, 0x78;" : "+r"(d) : "r"(z), "r"(zero)); }

// TIE 0: none, 1: every MMA, 2: every second MMA
template <int LAG, int TIE> struct EL : MLW {
  static_assert(LAG >= 0 && LAG < MI * NI, "an absorb runs before its accumulator's next MMA");
  using MLW::MLW;
  template <int J> DEV static void ab(St (&ch)[8], const int (&acc)[MI][NI][4]) { absorb<DUO, J / NI, J % NI>(ch, acc[J / NI][J % NI]); }
  template <int J> DEV static void flush(const int (&acc)[MI][NI][4], St (&ch)[8]) {
    if constexpr (J < MI * NI) {
      ab<J>(ch, acc);
      flush<J + 1>(acc, ch);
    }
  }
  // one atom; after MMA IDX, the absorb of tile IDX - LAG (of the previous atom's when negative; none in a tile's first)
  template <bool R, bool FIRST, int IDX = 0>
  DEV static void atom(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&fa)[MI][4], const uint32_t (&fb)[NI][2], uint32_t zero) {
    if constexpr (IDX < MI * NI) {
      constexpr int mi = IDX / NI, ni = IDX % NI, P = IDX - LAG;
      if constexpr (FIRST) {
        imma0(acc[mi][ni], fa[mi], fb[ni]);
      } else {
        if constexpr (R && (TIE == 1 || (TIE == 2 && IDX % 2 == 0))) {
          constexpr int TP = (IDX - 1 - LAG + 2 * MI * NI) % (MI * NI);   // the tile the previous absorb took
          tie(acc[mi][ni][0], ch[4 * (TP % 2) + 3].z, zero);
        }
        imma(acc[mi][ni], fa[mi], fb[ni]);
      }
      if constexpr (R && P >= 0) ab<P>(ch, acc);
      else if constexpr (R && !FIRST) ab<P + MI * NI>(ch, acc);
      atom<R, FIRST, IDX + 1>(acc, ch, fa, fb, zero);
    }
  }
  template <bool R, bool FIRST, bool LAST>
  DEV void stage_il(int (&acc)[MI][NI][4], St (&ch)[8], uint32_t (&fa)[2][MI][4], uint32_t (&fb)[2][NI][2], uint32_t zero) {
    const uint32_t sb = stage(s), done = empty(s);
    load(fa[1], fb[1], sb, 1);
    atom<R, FIRST>(acc, ch, fa[0], fb[0], zero);
    load(fa[0], fb[0], sb, 2);
    atom<R, false>(acc, ch, fa[1], fb[1], zero);
    load(fa[1], fb[1], sb, 3);
    atom<R, false>(acc, ch, fa[0], fb[0], zero);
    advance();
    if (!LAST) {
      mbar_wait_warp(full(s), ph);
      load(fa[0], fb[0], stage(s), 0);
    }
    atom<R, false>(acc, ch, fa[1], fb[1], zero);
    release(done);
  }
  template <bool R> DEV void tile_il(int K, int (&acc)[MI][NI][4], St (&ch)[8], uint32_t zero) {
    const int KT = K / BK;
    uint32_t fa[2][MI][4], fb[2][NI][2];
    mbar_wait_warp(full(s), ph);
    load(fa[0], fb[0], stage(s), 0);
    if (KT == 1) {
      stage_il<R, true, true>(acc, ch, fa, fb, zero);
    } else {
      stage_il<R, true, false>(acc, ch, fa, fb, zero);
#pragma unroll 1
      for (int kt = 1; kt < KT - 1; kt++) stage_il<R, false, false>(acc, ch, fa, fb, zero);
      stage_il<R, false, true>(acc, ch, fa, fb, zero);
    }
    if constexpr (R) flush<MI * NI - LAG>(acc, ch);
  }
  // the tail stage's rows P_A = [S_hi | 0 | S_lo | 0], P_B = [U | 0 | U | 0]: atoms 0 and 2
  DEV void peel2(int (&acc)[MI][NI][4]) {
    mbar_wait_warp(full(s), ph);
    const uint32_t tb = stage(s);
    uint32_t fb[2][NI][2];
    load_b(fb[0], tb, 0);
    load_b(fb[1], tb, 2);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      uint32_t fa[2][4];
      load_a(fa[0], tb, 0, mi);
      load_a(fa[1], tb, 2, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int t1[4], t2[4];
        imma0(t1, fa[0], fb[0][ni]);
        imma0(t2, fa[1], fb[1][ni]);
#pragma unroll
        for (int e = 0; e < 4; e++)
          acc[mi][ni][e] = (int)((uint32_t)acc[mi][ni][e] - ((uint32_t)t1[e] << 8) - (uint32_t)t2[e]);
      }
    }
    release(empty(s));
    advance();
  }
};

template <bool R, int LAG, int TIE> DEV void e_body(const Args& a) {
  using L = EL<LAG, TIE>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps, a.M, a.N, a.K, a.group, true, 1);
  ml.pa_rows = a.pa_rows;
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int tiles_m = a.M / 128, tiles_n = a.N / 128;
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  const uint32_t zero = (uint32_t)a.KR;
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn, bi;
    ml.tile_of(t, bi, tm, tn);
    const uint32_t tile_id = tm * tiles_n + tn;
    St ch[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      ch[e].y = mix32(a.seed ^ mix32((tile_id * L::NCT + ctid) * 8 + e));
      ch[e].z = mix32(ch[e].y + 0x9E3779B9u);
    }
    int acc[MI][NI][4];
    ml.template tile_il<R>(a.K, acc, ch, zero);
    ml.peel2(acc);
    const int row0 = tm * 128 + wm * 32 + g, col0 = tn * 128 + wn * 32 + 2 * q;
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int* p = a.U + (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni;
        *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
        *reinterpret_cast<int2*>(p + (size_t)8 * a.N) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
      }
    if constexpr (R) {
      uint4* r = reinterpret_cast<uint4*>(a.R) + ((size_t)tile_id * L::NCT + ctid) * 2;
      r[0] = make_uint4(finish<DUO>(ch, 0), finish<DUO>(ch, 1), finish<DUO>(ch, 2), finish<DUO>(ch, 3));
      r[1] = make_uint4(finish<DUO>(ch, 4), finish<DUO>(ch, 5), finish<DUO>(ch, 6), finish<DUO>(ch, 7));
    }
  }
}
#define EGEMM(NAME, R, LAG, TIE) \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ Args a) { e_body<R, LAG, TIE>(a); }
EGEMM(e_plain, false, 0, 0)
EGEMM(e_l0, true, 0, 0)
EGEMM(e_l0t, true, 0, 1)
EGEMM(e_l1, true, 1, 0)
EGEMM(e_l1t, true, 1, 1)
EGEMM(e_l2t, true, 2, 1)
EGEMM(e_l4t, true, 4, 1)
EGEMM(e_l1h, true, 1, 2)
EGEMM(e_l2h, true, 2, 2)
EGEMM(e_l3h, true, 3, 2)

// ---------------------------------------------------------------- the operands
// x (rows x k bf16) to X (rows x k int8): a warp a group of 4 rows, its lanes 8 columns each (512 contiguous bytes of x a
// step); DITHER: X = clamp(floor(x inv_s), -127, 126) + b and P_A's rows [S_hi | 0 | S_lo | 0] (lane c' of the warp ends
// with S at c = 8 (c' % 4) + c' / 4 after a reduce-scatter over the 8 lanes holding that column's partial sums), else
// X = clamp(rint(x inv_s), -127, 127)
template <bool DITHER>
__global__ void __launch_bounds__(256) quant(const uint16_t* __restrict__ x, int8_t* __restrict__ X, int8_t* __restrict__ PA,
                                             int rows, int k, float inv_s, const __grid_constant__ Sig sg) {
  __shared__ uint32_t dg[DITHER ? 8 : 1][DITHER ? 32 * 16 : 1];   // a warp's 32 patch digests, digest byte order
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, grp = blockIdx.x * 8 + warp;
  if (grp * 4 >= rows) return;
  const int kb = k / 128, steps = k / 256;
  if constexpr (DITHER) {
    if (lane < kb) {
      uint64_t st[8];
      sha_block(sg.w, (uint64_t)DITHER_TAG << 56 | ((uint64_t)grp * kb + lane), st);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        dg[warp][16 * lane + 2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
        dg[warp][16 * lane + 2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
      }
    }
    __syncwarp();
  }
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dg[DITHER ? warp : 0]);
#pragma unroll 1
  for (int rr = 0; rr < 4; rr++) {
    const size_t r = (size_t)grp * 4 + rr;
    const uint4* src = reinterpret_cast<const uint4*>(x + r * k);
    uint2* dst = reinterpret_cast<uint2*>(X + r * k);
    int v[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // S's partial sums at c = 8 (lane % 4) + i
#pragma unroll 4
    for (int it = 0; it < steps; it++) {
      const int ch = it * 32 + lane;   // columns 8 ch .. 8 ch + 7
      const uint4 w = __ldg(src + ch);
      uint32_t bits = 0;
      if constexpr (DITHER) bits = bytes[64 * (ch / 16) + 16 * rr + ch % 16];
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      int qv[8];
#pragma unroll
      for (int h = 0; h < 4; h++) {
        const float f0 = __uint_as_float(ww[h] << 16) * inv_s, f1 = __uint_as_float(ww[h] & 0xFFFF0000u) * inv_s;
        if constexpr (DITHER) {
          qv[2 * h] = min(max(__float2int_rd(f0), -127), 126) + (int)((bits >> (2 * h)) & 1u);
          qv[2 * h + 1] = min(max(__float2int_rd(f1), -127), 126) + (int)((bits >> (2 * h + 1)) & 1u);
        } else {
          qv[2 * h] = min(max(__float2int_rn(f0), -127), 127);
          qv[2 * h + 1] = min(max(__float2int_rn(f1), -127), 127);
        }
      }
      if constexpr (DITHER) {
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] += qv[i];
      }
      const uint32_t lo = __byte_perm(__byte_perm(qv[0], qv[1], 0x0040), __byte_perm(qv[2], qv[3], 0x0040), 0x5410);
      const uint32_t hi = __byte_perm(__byte_perm(qv[4], qv[5], 0x0040), __byte_perm(qv[6], qv[7], 0x0040), 0x5410);
      dst[ch] = make_uint2(lo, hi);
    }
    if constexpr (DITHER) {
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
      const int c = 8 * (lane & 3) + (lane >> 2), S = v[0], slo = (int)(int8_t)(S & 0xFF), shi = (S - slo) / 256;
      int8_t* pr = PA + r * 128;
      pr[c] = (int8_t)shi; pr[32 + c] = 0; pr[64 + c] = (int8_t)slo; pr[96 + c] = 0;
    }
  }
}

// W' = W + U (W'_jk = W_jk + U_{j, k mod 32}) and P_B's rows [U | 0 | U | 0]: a block 8 rows, its threads loading W's
// chunks while threads 0..3 hash U's blocks (rows 2j, 2j + 1); k <= 4096
__global__ void __launch_bounds__(256) wedit(const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp, uint8_t* __restrict__ PB,
                                             int rows, int k, const __grid_constant__ Sig sg) {
  __shared__ uint32_t us[8][8];
  const int r0 = blockIdx.x * 8, t = threadIdx.x, cpr = k / 16, total = 8 * cpr;
  constexpr int MAXC = 8;
  uint4 w[MAXC];
  const uint4* src = reinterpret_cast<const uint4*>(W + (size_t)r0 * k);
#pragma unroll
  for (int i = 0; i < MAXC; i++)
    if (t + 256 * i < total) w[i] = __ldg(src + t + 256 * i);
  if (t < 4) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)U_TAG << 56 | (uint64_t)(r0 / 2 + t), st);
#pragma unroll
    for (int rr = 0; rr < 2; rr++)
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
          const int e = 32 * rr + 4 * q + b;
          v |= u_byte((uint32_t)(st[e / 8] >> (56 - 8 * (e % 8))) & 0xFFu) << (8 * b);
        }
        us[2 * t + rr][q] = v;
      }
  }
  __syncthreads();
  {
    const int rr = t / 32, wq = t % 32;
    reinterpret_cast<uint32_t*>(PB + (size_t)(r0 + rr) * 128)[wq] = (wq / 8) % 2 ? 0u : us[rr][wq % 8];
  }
  uint4* dst = reinterpret_cast<uint4*>(Wp + (size_t)r0 * k);
#pragma unroll
  for (int i = 0; i < MAXC; i++) {
    const int c = t + 256 * i;
    if (c < total) {
      const int rr = c / cpr, u0 = (4 * (c % cpr)) % 8;
      const uint4 x = w[i];
      dst[c] = make_uint4(add_noise<false>(x.x, us[rr][u0]), add_noise<false>(x.y, us[rr][u0 + 1]), add_noise<false>(x.z, us[rr][u0 + 2]),
                          add_noise<false>(x.w, us[rr][u0 + 3]));
    }
  }
}

// x in bf16, (h / 2^24 * 2 - 1) 127.5 rounded to bf16; the gate's rows 0 and 1 at +200 and -200 (clamped)
__global__ void __launch_bounds__(256) xfill(uint16_t* x, size_t count, uint32_t seed, size_t k, int ends) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)i) ^ mix32((uint32_t)(i >> 32) + 0x77u));
    float f = ((float)(h & 0xFFFFFF) / 16777216.f * 2.f - 1.f) * 127.5f;
    if (ends && i < k) f = 200.f;
    else if (ends && i < 2 * k) f = -200.f;
    const uint32_t b = __float_as_uint(f);
    x[i] = (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
  }
}
__global__ void cmp_words(const uint32_t* a, const uint32_t* b, size_t n, unsigned long long* bad) {
  unsigned long long c = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += a[i] != b[i];
  if (c) atomicAdd(bad, c);
}

// ---------------------------------------------------------------- host
struct EP {
  int m, n, k, grid;
  uint16_t* x;
  int8_t *X, *Xq, *W, *Wp, *PA, *PB;
  int *Y, *Yr;
  uint32_t *R, *R2;
  size_t ny, nr;
  LtG plain, plainq;
  Maps* mE;
  uint32_t seed, inst = 0;
  float inv_s = 1.f;
  cudaStream_t st;
  unsigned long long* bad;
};

static const std::pair<const char*, Kern> EK[] = {{"e_plain", e_plain}, {"e_l0", e_l0},   {"e_l0t", e_l0t},    {"e_l1", e_l1},
                                                  {"e_l1t", e_l1t},     {"e_l2t", e_l2t}, {"e_l4t", e_l4t},    {"e_l1h", e_l1h},
                                                  {"e_l2h", e_l2h},     {"e_l3h", e_l3h}, {"t_duo32", t_duo32}};
static Kern ekern(const std::string& n) {
  for (auto& [name, k] : EK) if (n == name) return k;
  fprintf(stderr, "no kernel %s\n", n.c_str());
  exit(2);
}

static void esetup(EP& P, int m, int n, int k, uint32_t seed, bool gate) {
  P.m = m; P.n = n; P.k = k; P.seed = seed;
  if (m % 128 || n % 128 || k % 256 || k > 4096) { fprintf(stderr, "shape %d %d %d: 128-tiled, k %% 256 == 0, k <= 4096\n", m, n, k); exit(2); }
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  P.grid = std::min((m / 128) * (n / 128), nsm);
  CK(cudaStreamCreate(&P.st));
  P.ny = (size_t)m * n;
  P.nr = (size_t)(m / 128) * (n / 128) * 512 * 8;
  CK(cudaMalloc(&P.x, (size_t)m * k * 2));
  CK(cudaMalloc(&P.X, (size_t)m * k)); CK(cudaMalloc(&P.Xq, (size_t)m * k));
  CK(cudaMalloc(&P.W, (size_t)n * k)); CK(cudaMalloc(&P.Wp, (size_t)n * k));
  CK(cudaMalloc(&P.PA, (size_t)m * 128)); CK(cudaMalloc(&P.PB, (size_t)n * 128));
  CK(cudaMalloc(&P.Y, P.ny * 4)); CK(cudaMalloc(&P.Yr, P.ny * 4));
  CK(cudaMalloc(&P.R, P.nr * 4)); CK(cudaMalloc(&P.R2, P.nr * 4));
  CK(cudaMalloc(&P.bad, 8));
  xfill<<<4096, 256, 0, P.st>>>(P.x, (size_t)m * k, mix32(seed ^ 0x3333u), k, gate);
  sfill<I8><<<4096, 256, 0, P.st>>>(reinterpret_cast<uint8_t*>(P.W), (size_t)n * k, mix32(seed ^ 0x4444u), -63, 127);
  if (gate) { CK(cudaMemsetAsync(P.W, 0x3F, k, P.st)); CK(cudaMemsetAsync(P.W + k, 0xC1, k, P.st)); }   // rows at +63, -63
  quant<true><<<(m / 4 + 7) / 8, 256, 0, P.st>>>(P.x, P.X, P.PA, m, k, P.inv_s, sigma(seed, 0xFFFFFFFFu));
  quant<false><<<(m / 4 + 7) / 8, 256, 0, P.st>>>(P.x, P.Xq, nullptr, m, k, P.inv_s, sigma(seed, 0xFFFFFFFFu));
  wedit<<<n / 8, 256, 0, P.st>>>(reinterpret_cast<const uint8_t*>(P.W), reinterpret_cast<uint8_t*>(P.Wp), reinterpret_cast<uint8_t*>(P.PB),
                                  n, k, sigma(seed, 0xFFFFFFFFu));
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  P.plain = ltg(I8, m, n, k, k, k, n, P.X, P.W, P.Yr, 0, P.st);
  P.plainq = ltg(I8, m, n, k, k, k, n, P.Xq, P.W, P.Yr, 0, P.st);
  P.mE = upload_maps(Maps{tmap(P.X, k, m, 128), tmap(P.Wp, k, n, 128), tmap(P.PA, 128, m, 128), tmap(P.PB, 128, n, 128)});
  for (auto& [name, kern] : EK) CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
}
static void efree(EP& P) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)P.x, (void*)P.X, (void*)P.Xq, (void*)P.W, (void*)P.Wp, (void*)P.PA, (void*)P.PB, (void*)P.Y, (void*)P.Yr,
                  (void*)P.R, (void*)P.R2, (void*)P.bad, (void*)P.mE})
    CK(cudaFree(p));
  CK(cudaStreamDestroy(P.st));
}
static void egemm_run(EP& P, Kern kern, int* Y, uint32_t* R) {
  Args a{P.mE, Y, R, P.m, P.n, P.k, 8, P.seed, 0, 1, 0, 0, {0, 0, 0, 0}, nullptr, 0};
  kern<<<P.grid, MLW::NT, CSMEM, P.st>>>(a);
}
static void qd(EP& P, const Sig& s) { quant<true><<<(P.m / 4 + 7) / 8, 256, 0, P.st>>>(P.x, P.X, P.PA, P.m, P.k, P.inv_s, s); }
static void qp(EP& P, const Sig& s) { quant<false><<<(P.m / 4 + 7) / 8, 256, 0, P.st>>>(P.x, P.Xq, nullptr, P.m, P.k, P.inv_s, s); }
static void we(EP& P, const Sig& s) {
  wedit<<<P.n / 8, 256, 0, P.st>>>(reinterpret_cast<const uint8_t*>(P.W), reinterpret_cast<uint8_t*>(P.Wp),
                                    reinterpret_cast<uint8_t*>(P.PB), P.n, P.k, s);
}

// an arm: plain (cuBLASLt on X, W), plainq (the plain quantizer, then cuBLASLt on its X), qplain, qdither, wedit, a
// GEMM kernel by name (EK), edited:KERN (wedit, then KERN), editedq:KERN (qdither, wedit, KERN: one key for both)
static std::function<void(EP&)> earm(const std::string& a) {
  if (a == "plain") return [](EP& P) { ltg_run(P.plain, P.st); };
  if (a == "plainq") return [](EP& P) { qp(P, sigma(P.seed, P.inst++)); ltg_run(P.plainq, P.st); };
  if (a == "qplain") return [](EP& P) { qp(P, sigma(P.seed, P.inst++)); };
  if (a == "qdither") return [](EP& P) { qd(P, sigma(P.seed, P.inst++)); };
  if (a == "wedit") return [](EP& P) { we(P, sigma(P.seed, P.inst++)); };
  if (a.rfind("edited:", 0) == 0) {
    Kern k = ekern(a.substr(7));
    return [k](EP& P) { we(P, sigma(P.seed, P.inst++)); egemm_run(P, k, P.Y, P.R); };
  }
  if (a.rfind("editedq:", 0) == 0) {
    Kern k = ekern(a.substr(8));
    return [k](EP& P) {
      const Sig s = sigma(P.seed, P.inst++);
      qd(P, s); we(P, s); egemm_run(P, k, P.Y, P.R);
    };
  }
  Kern k = ekern(a);
  return [k](EP& P) { egemm_run(P, k, P.Y, P.R); };
}

static unsigned long long ecmp(EP& P, const void* a, const void* b, size_t words) {
  CK(cudaMemsetAsync(P.bad, 0, 8, P.st));
  cmp_words<<<1024, 256, 0, P.st>>>((const uint32_t*)a, (const uint32_t*)b, words, P.bad);
  unsigned long long h;
  CK(cudaMemcpyAsync(&h, P.bad, 8, cudaMemcpyDeviceToHost, P.st));
  CK(cudaStreamSynchronize(P.st));
  return h;
}

static int egate(const std::string& dir, int m, int n, int k, uint32_t seed) {
  EP P;
  esetup(P, m, n, k, seed, true);
  const size_t xb = (size_t)m * k, wb = (size_t)n * k;
  auto poison = [&] {
    for (auto [p, b] : {std::pair<void*, size_t>{P.X, xb}, {P.Xq, xb}, {P.Wp, wb}, {P.PA, (size_t)m * 128}, {P.PB, (size_t)n * 128},
                        {P.Y, P.ny * 4}, {P.Yr, P.ny * 4}, {P.R, P.nr * 4}, {P.R2, P.nr * 4}})
      CK(cudaMemsetAsync(p, 0xA5, b, P.st));
  };
  poison();
  const Sig s = sigma(seed, 0);
  qd(P, s);
  we(P, s);
  qp(P, s);
  ltg_run(P.plain, P.st);
  egemm_run(P, t_duo32, P.Y, P.R2);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  bool ok = true;
  std::string res;
  auto rec = [&](const std::string& name, unsigned long long y, long long r) {
    res += (res.empty() ? "" : ", ") + std::string("\"") + name + "\": {\"y_bad\": " + std::to_string(y) + ", \"r_bad\": " + std::to_string(r) + "}";
  };
  const unsigned long long ty = ecmp(P, P.Y, P.Yr, P.ny);
  rec("t_duo32", ty, -1);
  ok &= ty == 0;
  dump(dir, "R_t_duo32_tile0.bin", P.R2, 512 * 8 * 4, true);
  for (auto& [name, kern] : EK) {
    if (kern == t_duo32) continue;
    CK(cudaMemsetAsync(P.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(P.R, 0xA5, P.nr * 4, P.st));
    egemm_run(P, kern, P.Y, P.R);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    const bool rcp = std::string(name) != "e_plain";
    const unsigned long long y = ecmp(P, P.Y, P.Yr, P.ny), r = rcp ? ecmp(P, P.R, P.R2, P.nr) : 0;
    rec(name, y, rcp ? (long long)r : -1);
    ok &= y == 0 && r == 0;
  }
  // the no-write control: poisoned, nothing run; both compares must fail
  CK(cudaMemsetAsync(P.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(P.R, 0xA5, P.nr * 4, P.st));
  const unsigned long long cy = ecmp(P, P.Y, P.Yr, P.ny), cr = ecmp(P, P.R, P.R2, P.nr);
  const bool control_rejected = cy > 0 && cr > 0;
  rec("control", cy, (long long)cr);
  // the tiles egemm_ref.py recomputes from x, W and the key: e_l1t's output and receipts (t_duo32's equal them on device)
  egemm_run(P, e_l1t, P.Y, P.R);
  CK(cudaStreamSynchronize(P.st));
  const int tiles_m = m / 128, tiles_n = n / 128;
  const int tl[3][2] = {{0, 0}, {tiles_m / 2, tiles_n / 2}, {tiles_m - 1, tiles_n - 1}};
  std::string tj;
  for (int i = 0; i < 3; i++) {
    const int tm = tl[i][0], tn = tl[i][1];
    const std::string sfx = "_" + std::to_string(tm) + "_" + std::to_string(tn) + ".bin";
    dump(dir, ("x" + sfx).c_str(), P.x + (size_t)tm * 128 * k, 128 * (size_t)k * 2, true);
    dump(dir, ("X" + sfx).c_str(), P.X + (size_t)tm * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("Xq" + sfx).c_str(), P.Xq + (size_t)tm * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("PA" + sfx).c_str(), P.PA + (size_t)tm * 128 * 128, 128 * 128, true);
    dump(dir, ("W" + sfx).c_str(), P.W + (size_t)tn * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("Wp" + sfx).c_str(), P.Wp + (size_t)tn * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("PB" + sfx).c_str(), P.PB + (size_t)tn * 128 * 128, 128 * 128, true);
    std::vector<int> yt(128 * 128);
    for (int r = 0; r < 128; r++)
      CK(cudaMemcpy(yt.data() + r * 128, P.Y + (size_t)(tm * 128 + r) * n + tn * 128, 128 * 4, cudaMemcpyDeviceToHost));
    dump(dir, ("Y" + sfx).c_str(), yt.data(), yt.size() * 4, false);
    dump(dir, ("R" + sfx).c_str(), P.R + (size_t)(tm * tiles_n + tn) * 512 * 8, 512 * 8 * 4, true);
    tj += std::string(i ? ", " : "") + "[" + std::to_string(tm) + ", " + std::to_string(tn) + "]";
  }
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"inv_s\": %.9g, \"grid\": %d, \"key\": [\"%016llx\", \"%016llx\", "
              "\"%016llx\", \"%016llx\"], \"tiles\": [%s], \"device\": {%s}, \"device_ok\": %s, \"control_rejected\": %s}\n",
          m, n, k, seed, P.inv_s, P.grid, (unsigned long long)s.w[0], (unsigned long long)s.w[1], (unsigned long long)s.w[2],
          (unsigned long long)s.w[3], tj.c_str(), res.c_str(), ok ? "true" : "false", control_rejected ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"device_ok\": %s, \"control_rejected\": %s, \"device\": {%s}}\n",
         dir.c_str(), m, n, k, ok ? "true" : "false", control_rejected ? "true" : "false", res.c_str());
  efree(P);
  return ok && control_rejected ? 0 : 1;
}

static std::vector<std::string> split(const char* s, const char* dflt) {
  std::vector<std::string> v;
  std::string a = s ? s : dflt, cur;
  for (char c : a + ",") {
    if (c == ',') { if (!cur.empty()) v.push_back(cur); cur.clear(); }
    else cur += c;
  }
  return v;
}

static int ebench(int reps, int m, int n, int k) {
  Nv nv;
  const std::vector<std::string> names = split(getenv("EG_ARMS"), "plain,plainq,qplain,qdither,wedit,e_plain,t_duo32,e_l0,e_l1t,e_l2t");
  EP P;
  esetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
  std::vector<std::function<void(EP&)>> arms;
  for (auto& a : names) arms.push_back(earm(a));
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
  printf("{\"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d, \"lt_algo\": %d, \"arms\": {", m, n, k, P.grid, P.plain.algo_id);
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
  efree(P);
  return 0;
}

// per call, the slice's way: ARM and BASE alternate which goes first, each call timed by its own events
static int eil(int reps, const std::string& arm, const std::string& base, int m, int n, int k) {
  Nv nv;
  EP P;
  esetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
  auto fa = earm(arm), fb = earm(base);
  cudaEvent_t e[4];
  for (auto& x : e) CK(cudaEventCreate(&x));
  std::vector<double> ta, tb, mz;
  for (int r = -5; r < reps; r++) {
    const bool afirst = (r & 1) != 0;
    CK(cudaEventRecord(e[0], P.st));
    (afirst ? fa : fb)(P);
    CK(cudaEventRecord(e[1], P.st));
    (afirst ? fb : fa)(P);
    CK(cudaEventRecord(e[2], P.st));
    CK(cudaEventSynchronize(e[2]));
    float t0, t1;
    CK(cudaEventElapsedTime(&t0, e[0], e[1]));
    CK(cudaEventElapsedTime(&t1, e[1], e[2]));
    if (r < 0) continue;
    ta.push_back(afirst ? t0 : t1);
    tb.push_back(afirst ? t1 : t0);
    mz.push_back(nv.sm());
  }
  CK(cudaGetLastError());
  const double a = median(ta), b = median(tb);
  printf("{\"il\": \"%s\", \"base\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"reps\": %d, \"ms\": %.6f, \"base_ms\": %.6f, "
         "\"ratio\": %.4f, \"mhz\": %.0f, \"mhz_min\": %.0f, \"mhz_max\": %.0f}\n",
         arm.c_str(), base.c_str(), m, n, k, reps, a, b, a / b, median(mz), *std::min_element(mz.begin(), mz.end()),
         *std::max_element(mz.begin(), mz.end()));
  fflush(stdout);
  for (auto& x : e) CK(cudaEventDestroy(x));
  efree(P);
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return egate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 6 && !strcmp(argv[1], "bench")) return ebench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 8 && !strcmp(argv[1], "il")) return eil(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
  fprintf(stderr, "usage: egemm gate DIR M N K SEED | bench REPS M N K | il REPS ARM BASE M N K\n");
  return 2;
}
