// The served useful GEMM on the exact int8 grid, with and without receipts of its running sums, on sm_120a (RTX PRO 6000).
//
// The scheme (note:20261015T0727Z-draft-int8-grid): Q(A) (m x k) and Q(B) (n x k) are row-major int8 codes in [-121, 121].
// Noise per chunk of L = 32 columns: A~ = Q(A) + E_A F_A^T, B~ = Q(B) + E_B F_B^T, with E_A (m x 32), E_B (n x 32) uniform on
// {-3..3} (the live coins) and F_A, F_B (k x 32) two signed permutations per chunk (fixed per shape). The tensor cores run
// C = A~ B~^T (s8 x s8 -> s32, mma.sync m16n8k32), and the exact peel returns
//   U = Q(A) Q(B)^T = C - E_A (B~ F_A)^T - (A~ F_B - E_A G) E_B^T,   G = F_A^T F_B (32 x 32),
// with the two wide factors W = B~ F_A and V = A~ F_B - E_A G in two int8 limbs (256 hi + lo), so the peel is four more k = 32
// atoms a tile (4L/k of the product), run in the mainloop's tail stage.
//
// Receipts (Pi_L): every RS = 32 or 64 steps of k, each accumulator word T_s (the running sum C over the first s columns)
// is absorbed into a per-thread chain; the chains' finishes are the tile's receipts. phi = duo (3 SASS ops a word,
// note:20261015T0432Z-draft-receipt-phi: u = y ^ p ^ k_t; y = u a_t + z; z = rotl(u, r_t) + y; receipt y ^ rotl(z, 16)) or
// duo2 (two duo lanes a word with their own keys and rotations, note:20261015T0711Z-draft-receipt-phi-followup, about 6 ops).
// Chain e of consumer thread (warp w, lane) absorbs, at each receipt boundary in ascending k, the words acc[mi][ni][e] in
// (mi, ni) order, at key position t = mi NI + ni. The key schedule (16 positions) is a cost stand-in for the reviewed one.
//
// Kernels: form (A~ or B~ and the tail rows P_A = [E_A | V_hi | E_A | V_lo], P_B = [W_hi | E_B | W_lo | E_B], one warp a row),
// and the GEMM (a TMA producer warpgroup and 8 consumer warps, mainloop_sm120.cuh's schedule with s32 accumulators).
// cuBLASLt's int8 GEMM (s32 out) is the baseline. Modes:
//   fgemm gate DIR M N K GRID SEED P|D  every arm on one small shape at tile config P or D, inputs and outputs to DIR for ref.py
//   fgemm bench REPS SHAPE...           interleaved timing, one JSON line a shape (SHAPE = name:m:n:k)
//   fgemm one ARM M N K                 warm-up then one call of ARM (for ncu)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a fgemm.cu -lcuda -lcublasLt -lnvidia-ml
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));     \
      exit(3);                                                                                 \
    }                                                                                          \
  } while (0)
#define CL(x)                                                                                  \
  do {                                                                                         \
    cublasStatus_t s_ = (x);                                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                         \
      fprintf(stderr, "%s:%d %s: cublas %d\n", __FILE__, __LINE__, #x, int(s_));              \
      exit(3);                                                                                 \
    }                                                                                          \
  } while (0)
#define DEV __device__ __forceinline__

__host__ __device__ inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}

// ---------------------------------------------------------------- receipts
enum Kind { NONE = 0, DUO = 1, DUO2 = 2 };
constexpr int NPOS = 16;
__constant__ uint32_t KA[2][NPOS], KK[2][NPOS];
__host__ __device__ constexpr int rot_of(int lane, int t) { return lane == 0 ? ((t * 7 + 5) % 31) + 1 : ((t * 11 + 3) % 29) + 1; }
struct St { uint32_t y, z; };
DEV uint32_t rotl(uint32_t x, int r) { return __funnelshift_l(x, x, r); }
template <int LANE, int T> DEV void duo(St& s, uint32_t p) {
  const uint32_t u = s.y ^ p ^ KK[LANE][T];
  s.y = u * KA[LANE][T] + s.z;
  s.z = rotl(u, rot_of(LANE, T)) + s.y;
}
template <int KIND> constexpr int NCH = KIND == DUO2 ? 8 : 4;

// ---------------------------------------------------------------- TMA, mbarrier, ldmatrix, mma.sync
DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEV void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
// a consumer's release: its ldmatrix reads (generic proxy) are ordered before the TMA refill (async proxy) by the fence
DEV void mbar_release(uint32_t bar) {
  asm volatile("fence.proxy.async.shared::cta;\n mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
DEV void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n FG_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra FG_WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
DEV void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
DEV void tma_acquire(const void* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"(map) : "memory");
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}
DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
template <int R> DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
DEV void ldsm4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// d += a b, one s8 atom (k = 32): A rows g, g + 8 x k 4t.., 16 + 4t..; B column g x k 4t.., 16 + 4t..; D rows g, g + 8 x
// columns 2t, 2t + 1 (g = lane / 4, t = lane % 4)
DEV void imma(int* d, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEV void imma0(int* d, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0));
}
DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

struct Maps { CUtensorMap a, b, pa, pb; };
struct Args {
  const Maps* maps;
  int* U;            // m x n s32, row-major
  uint32_t* R;       // receipts: [tile][consumer thread][NCH]
  int M, N, K, group;
  uint32_t seed;
};

// TM x BN tile, 8 consumer warps of WM x 32 (MI = WM / 16 row blocks of 16, NI = 4 column blocks of 8), one producer
// warpgroup whose first lane issues every TMA; STAGES stages of 128 bytes of depth (4 atoms); TAIL: the peel rows' stage.
template <int TM, int BN, int WM, int STAGES, int RP = 40> struct ML {
  static constexpr int BK = 128, WN = 32, WARPS_M = TM / WM, WARPS_N = BN / WN, NCW = WARPS_M * WARPS_N, NCT = NCW * 32,
                       MI = WM / 16, NI = WN / 8;
  static_assert(NCW == 8, "8 consumer warps");
  static constexpr int NT = NCT + 128, REGS_P = RP, REGS_LAUNCH = 65536 / NT / 8 * 8,
                       REGS_FIT = (NT * REGS_LAUNCH - 128 * REGS_P) / NCT / 8 * 8, REGS_C = REGS_FIT < 256 ? REGS_FIT : 256;
  static constexpr uint32_t A_BYTES = TM * BK, STAGE = (TM + BN) * BK;
  static constexpr int SMEM = STAGES * STAGE + 16 * STAGES + 1024;
  static_assert(STAGE % 1024 == 0 && A_BYTES % 1024 == 0, "boxes 1,024-byte aligned for SWIZZLE_128B");

  uint32_t base, bars;
  int s = 0;
  uint32_t ph = 0;
  uint32_t a_off[4], b_off[4];

  DEV explicit ML(uint8_t* smem) {
    base = (smem_u32(smem) + 1023) & ~1023u;
    bars = base + STAGES * STAGE;
  }
  DEV uint32_t full(int i) const { return bars + 8 * i; }
  DEV uint32_t empty(int i) const { return bars + 8 * (STAGES + i); }
  DEV uint32_t stage(int i) const { return base + i * STAGE; }
  DEV void advance() {
    if (++s == STAGES) { s = 0; ph ^= 1; }
  }
  DEV void init(int tid) {
    if (tid == 0) {
      for (int i = 0; i < STAGES; i++) { mbar_init(full(i), 1); mbar_init(empty(i), NCT); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
  }
  template <bool TAIL> DEV void produce(const Maps* maps, int M, int N, int K, int group) {
    const int tiles_m = M / TM, tiles_n = N / BN, KT = K / BK;
    tma_acquire(&maps->a); tma_acquire(&maps->b);
    if (TAIL) { tma_acquire(&maps->pa); tma_acquire(&maps->pb); }
    for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
      int tm, tn;
      raster(t, tiles_m, tiles_n, group, tm, tn);
      for (int kt = 0; kt < KT; kt++) {
        mbar_wait(empty(s), ph ^ 1);
        mbar_expect_tx(full(s), STAGE);
        tma_load(stage(s), &maps->a, full(s), kt * BK, tm * TM);
        tma_load(stage(s) + A_BYTES, &maps->b, full(s), kt * BK, tn * BN);
        advance();
      }
      if (TAIL) {
        mbar_wait(empty(s), ph ^ 1);
        mbar_expect_tx(full(s), STAGE);
        tma_load(stage(s), &maps->pa, full(s), 0, tm * TM);
        tma_load(stage(s) + A_BYTES, &maps->pb, full(s), 0, tn * BN);
        advance();
      }
    }
  }
  DEV void consumer_init(int warp, int lane) {
    const int wm = warp / WARPS_N, wn = warp % WARPS_N, x = lane & 7;
    const int a_row = wm * WM + (lane & 15), a_half = lane >> 4;
    const int b_row = wn * WN + (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      a_off[kk] = a_row * BK + (((2 * kk + a_half) ^ x) << 4);
      b_off[kk] = A_BYTES + b_row * BK + (((2 * kk + b_half) ^ x) << 4);
    }
  }
  DEV void load_a(uint32_t (&fa)[4], uint32_t sb, int kk, int mi) const { ldsm4(fa, sb + a_off[kk] + mi * 16 * BK); }
  DEV void load_b(uint32_t (&fb)[NI][2], uint32_t sb, int kk) const {
#pragma unroll
    for (int nj = 0; nj < NI / 2; nj++) {
      uint32_t r[4];
      ldsm4(r, sb + b_off[kk] + nj * 16 * BK);
      fb[2 * nj][0] = r[0]; fb[2 * nj][1] = r[1]; fb[2 * nj + 1][0] = r[2]; fb[2 * nj + 1][1] = r[3];
    }
  }
  DEV void load(uint32_t (&fa)[MI][4], uint32_t (&fb)[NI][2], uint32_t sb, int kk) const {
#pragma unroll
    for (int mi = 0; mi < MI; mi++) load_a(fa[mi], sb, kk, mi);
    load_b(fb, sb, kk);
  }
  // absorb acc[mi][ni]'s four words into chains 0-3 (and the second lane's 4-7) at position mi NI + ni
  template <int KIND, int MI_, int NI_> DEV static void absorb(St (&ch)[NCH<KIND>], const int (&w)[4]) {
    constexpr int T = MI_ * NI + NI_;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      duo<0, T>(ch[e], (uint32_t)w[e]);
      if constexpr (KIND == DUO2) duo<1, T>(ch[4 + e], (uint32_t)w[e]);
    }
  }
  template <int KIND, int IDX> DEV static void absorb_at(St (&ch)[NCH<KIND>], const int (&w)[4]) {
    absorb<KIND, IDX / NI, IDX % NI>(ch, w);
  }
  template <int KIND, int IDX = 0> DEV static void absorb_all(St (&ch)[NCH<KIND>], int (&acc)[MI][NI][4]) {
    if constexpr (IDX < MI * NI) {
      absorb_at<KIND, IDX>(ch, acc[IDX / NI][IDX % NI]);
      absorb_all<KIND, IDX + 1>(ch, acc);
    }
  }
  // one atom: ABS, absorb each accumulator's running sum (the previous atom's result) just before its next MMA
  template <bool FROM0, bool ABS, int KIND, int IDX = 0>
  DEV static void mma(int (&acc)[MI][NI][4], St (&ch)[NCH<KIND>], const uint32_t (&fa)[MI][4], const uint32_t (&fb)[NI][2]) {
    if constexpr (IDX < MI * NI) {
      constexpr int mi = IDX / NI, ni = IDX % NI;
      if constexpr (ABS) absorb_at<KIND, IDX>(ch, acc[mi][ni]);
      if (FROM0) imma0(acc[mi][ni], fa[mi], fb[ni]);
      else imma(acc[mi][ni], fa[mi], fb[ni]);
      mma<FROM0, ABS, KIND, IDX + 1>(acc, ch, fa, fb);
    }
  }
  // atom j of a stage absorbs when a receipt boundary falls just before it (never before the tile's first atom)
  __host__ __device__ static constexpr bool absorbs(int rs, bool first, int j) { return rs != 0 && !(first && j == 0) && (rs == 32 || j % 2 == 0); }
  template <int RS, int KIND, bool FIRST, bool LAST>
  DEV void stage_body(int (&acc)[MI][NI][4], St (&ch)[NCH<KIND>], uint32_t (&fa)[2][MI][4], uint32_t (&fb)[2][NI][2]) {
    const uint32_t sb = stage(s), done = empty(s);
    load(fa[1], fb[1], sb, 1);
    mma<FIRST, absorbs(RS, FIRST, 0), KIND>(acc, ch, fa[0], fb[0]);
    load(fa[0], fb[0], sb, 2);
    mma<false, absorbs(RS, FIRST, 1), KIND>(acc, ch, fa[1], fb[1]);
    load(fa[1], fb[1], sb, 3);
    mma<false, absorbs(RS, FIRST, 2), KIND>(acc, ch, fa[0], fb[0]);
    advance();
    if (!LAST) {
      mbar_wait_warp(full(s), ph);
      load(fa[0], fb[0], stage(s), 0);
    }
    mma<false, absorbs(RS, FIRST, 3), KIND>(acc, ch, fa[1], fb[1]);
    mbar_release(done);
  }
  template <int RS, int KIND> DEV void tile(int K, int (&acc)[MI][NI][4], St (&ch)[NCH<KIND>]) {
    const int KT = K / BK;
    uint32_t fa[2][MI][4], fb[2][NI][2];
    mbar_wait_warp(full(s), ph);
    load(fa[0], fb[0], stage(s), 0);
    stage_body<RS, KIND, true, false>(acc, ch, fa, fb);
#pragma unroll 1
    for (int kt = 1; kt < KT - 1; kt++) stage_body<RS, KIND, false, false>(acc, ch, fa, fb);
    stage_body<RS, KIND, false, true>(acc, ch, fa, fb);
    if constexpr (RS != 0) absorb_all<KIND>(ch, acc);
  }
  // the peel: U = C - 256 (E_A W_hi^T + V_hi E_B^T) - (E_A W_lo^T + V_lo E_B^T), the tail stage's four atoms
  DEV void peel(int (&acc)[MI][NI][4]) {
    mbar_wait_warp(full(s), ph);
    const uint32_t tb = stage(s);
    uint32_t fb[4][NI][2];
#pragma unroll
    for (int kk = 0; kk < 4; kk++) load_b(fb[kk], tb, kk);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      uint32_t fa[4][4];
#pragma unroll
      for (int kk = 0; kk < 4; kk++) load_a(fa[kk], tb, kk, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int t1[4], t2[4];
        imma0(t1, fa[0], fb[0][ni]);
        imma(t1, fa[1], fb[1][ni]);
        imma0(t2, fa[2], fb[2][ni]);
        imma(t2, fa[3], fb[3][ni]);
#pragma unroll
        for (int e = 0; e < 4; e++)
          acc[mi][ni][e] = (int)((uint32_t)acc[mi][ni][e] - ((uint32_t)t1[e] << 8) - (uint32_t)t2[e]);
      }
    }
    mbar_release(empty(s));
    advance();
  }
};

template <int TM, int BN, int WM, int STAGES, bool TAIL, int RS, int KIND> DEV void gemm_body(const Args& a) {
  using L = ML<TM, BN, WM, STAGES>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem);
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp >= L::NCW) {
    regs_dec<L::REGS_P>();
    if (warp == L::NCW && lane == 0) ml.template produce<TAIL>(a.maps, a.M, a.N, a.K, a.group);
    asm volatile("exit;");
  }
  regs_inc<L::REGS_C>();
  ml.consumer_init(warp, lane);
  const int tiles_m = a.M / TM, tiles_n = a.N / BN;
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    raster(t, tiles_m, tiles_n, a.group, tm, tn);
    const uint32_t tile_id = tm * tiles_n + tn;
    St ch[NCH<KIND>];
#pragma unroll
    for (int e = 0; e < NCH<KIND>; e++) {
      ch[e].y = mix32(a.seed ^ mix32((tile_id * L::NCT + ctid) * NCH<KIND> + e));
      ch[e].z = mix32(ch[e].y + 0x9E3779B9u);
    }
    int acc[MI][NI][4];
    ml.template tile<RS, KIND>(a.K, acc, ch);
    if constexpr (TAIL) ml.peel(acc);
    const int row0 = tm * TM + wm * WM + g, col0 = tn * BN + wn * L::WN + 2 * q;
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int* p = a.U + (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni;
        *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
        *reinterpret_cast<int2*>(p + (size_t)8 * a.N) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
      }
    if constexpr (RS != 0) {
      uint32_t* r = a.R + ((size_t)tile_id * L::NCT + ctid) * NCH<KIND>;
#pragma unroll
      for (int e = 0; e < NCH<KIND>; e++) r[e] = ch[e].y ^ rotl(ch[e].z, 16);
    }
  }
}

// ---------------------------------------------------------------- the kernels
// configurations: P (prefill) 128 x 128 tiles, warps 64 x 32, 3 stages; D (decode) 128 x 32, warps 16 x 32, 4 stages
#define GEMM(NAME, TM, BN, WM, ST, TAIL, RS, KIND)                                                               \
  __global__ void __launch_bounds__(384, 1) NAME(const Args a) { gemm_body<TM, BN, WM, ST, TAIL, RS, KIND>(a); }
GEMM(p_plain, 128, 128, 64, 3, false, 0, NONE)
GEMM(p_g0, 128, 128, 64, 3, true, 0, NONE)
GEMM(p_duo32, 128, 128, 64, 3, true, 32, DUO)
GEMM(p_duo64, 128, 128, 64, 3, true, 64, DUO)
GEMM(p_duo2_32, 128, 128, 64, 3, true, 32, DUO2)
GEMM(p_duo2_64, 128, 128, 64, 3, true, 64, DUO2)
GEMM(d_plain, 128, 32, 16, 4, false, 0, NONE)
GEMM(d_g0, 128, 32, 16, 4, true, 0, NONE)
GEMM(d_duo32, 128, 32, 16, 4, true, 32, DUO)
GEMM(d_duo64, 128, 32, 16, 4, true, 64, DUO)
GEMM(d_duo2_32, 128, 32, 16, 4, true, 32, DUO2)
GEMM(d_duo2_64, 128, 32, 16, 4, true, 64, DUO2)

// Forming, one warp a row. own[col] = pi1 | pi2 << 8 | s1 << 16 | s2 << 17: the noise at column col of chunk s is
// (-1)^s1 E[r, pi1] + (-1)^s2 E[r, pi2] (F_own's row col). inv[s 32 + j] = c1 | c2 << 8 | s1 << 16 | s2 << 17 for the other
// side's F: its column j of chunk s has (-1)^s1 at row c1 and (-1)^s2 at row c2, so Y[r, j] = sum over chunks of
// (-1)^s1 X~[r, s 32 + c1] + (-1)^s2 X~[r, s 32 + c2] = (X~ F_other)[r, j]. Side A subtracts (E_A G)[r, j]. Y in two limbs
// (lo = the low byte as s8, hi = (Y - lo) / 256), and *flag |= 1 if hi leaves s8.
template <bool SIDE_A>
__global__ void __launch_bounds__(256) form_k(const int8_t* __restrict__ Q, const int8_t* __restrict__ E,
                                              const uint32_t* __restrict__ own, const uint32_t* __restrict__ inv,
                                              const int* __restrict__ G, int8_t* __restrict__ X, int8_t* __restrict__ P,
                                              int rows, int K, int* flag) {
  const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int e_l = E[(size_t)row * 32 + lane];
  const int8_t* q = Q + (size_t)row * K;
  int8_t* x = X + (size_t)row * K;
  int y = 0;
  for (int c0 = 0; c0 < K; c0 += 128) {
    const uint32_t qw = *reinterpret_cast<const uint32_t*>(q + c0 + 4 * lane);
    const uint4 d4 = *reinterpret_cast<const uint4*>(own + c0 + 4 * lane);
    const uint32_t d[4] = {d4.x, d4.y, d4.z, d4.w};
    uint32_t xw = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int e1 = __shfl_sync(0xFFFFFFFFu, e_l, d[i] & 31), e2 = __shfl_sync(0xFFFFFFFFu, e_l, (d[i] >> 8) & 31);
      const int v = (int)(int8_t)(qw >> (8 * i)) + ((d[i] >> 16) & 1 ? -e1 : e1) + ((d[i] >> 17) & 1 ? -e2 : e2);
      xw |= (uint32_t)(v & 0xFF) << (8 * i);
    }
    *reinterpret_cast<uint32_t*>(x + c0 + 4 * lane) = xw;
#pragma unroll
    for (int ch = 0; ch < 4; ch++) {
      const uint32_t di = inv[c0 + 32 * ch + lane];
      const int p1 = di & 31, p2 = (di >> 8) & 31;
      const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, xw, ch * 8 + (p1 >> 2)), w2 = __shfl_sync(0xFFFFFFFFu, xw, ch * 8 + (p2 >> 2));
      const int b1 = (int)(int8_t)(w1 >> (8 * (p1 & 3))), b2 = (int)(int8_t)(w2 >> (8 * (p2 & 3)));
      y += ((di >> 16) & 1 ? -b1 : b1) + ((di >> 17) & 1 ? -b2 : b2);
    }
  }
  if constexpr (SIDE_A) {
#pragma unroll 8
    for (int i = 0; i < 32; i++) y -= __shfl_sync(0xFFFFFFFFu, e_l, i) * G[i * 32 + lane];
  }
  const int lo = (int)(int8_t)(y & 0xFF), hi = (y - lo) >> 8;
  if (hi < -128 || hi > 127) atomicOr(flag, 1);
  int8_t* p = P + (size_t)row * 128;
  if (SIDE_A) {
    p[lane] = (int8_t)e_l; p[32 + lane] = (int8_t)hi; p[64 + lane] = (int8_t)e_l; p[96 + lane] = (int8_t)lo;
  } else {
    p[lane] = (int8_t)hi; p[32 + lane] = (int8_t)e_l; p[64 + lane] = (int8_t)lo; p[96 + lane] = (int8_t)e_l;
  }
}

// ---------------------------------------------------------------- host
struct Rng {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  int range(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }
};

// F (k x 32): two signed permutations a chunk, as own-side descriptors; inv the column view for the other side's forming
struct F {
  std::vector<uint32_t> own, inv;
  F(int K, Rng& r) : own(K), inv(K) {
    for (int s = 0; s < K / 32; s++) {
      int p1[32], p2[32];
      for (int i = 0; i < 32; i++) p1[i] = p2[i] = i;
      for (int i = 31; i > 0; i--) { std::swap(p1[i], p1[r.next() % (i + 1)]); std::swap(p2[i], p2[r.next() % (i + 1)]); }
      int c1[32], c2[32];
      uint32_t s1[32], s2[32];
      for (int c = 0; c < 32; c++) {
        s1[c] = r.next() & 1; s2[c] = r.next() & 1;
        own[s * 32 + c] = p1[c] | p2[c] << 8 | s1[c] << 16 | s2[c] << 17;
        c1[p1[c]] = c; c2[p2[c]] = c;
      }
      for (int j = 0; j < 32; j++) inv[s * 32 + j] = c1[j] | c2[j] << 8 | s1[c1[j]] << 16 | s2[c2[j]] << 17;
    }
  }
};
static std::vector<int> gram(const F& fa, const F& fb) {   // G = F_A^T F_B
  std::vector<int> g(32 * 32, 0);
  for (size_t col = 0; col < fa.own.size(); col++) {
    const uint32_t a = fa.own[col], b = fb.own[col];
    const int ai[2] = {(int)(a & 31), (int)((a >> 8) & 31)}, as[2] = {(a >> 16) & 1 ? -1 : 1, (a >> 17) & 1 ? -1 : 1};
    const int bi[2] = {(int)(b & 31), (int)((b >> 8) & 31)}, bs[2] = {(b >> 16) & 1 ? -1 : 1, (b >> 17) & 1 ? -1 : 1};
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) g[ai[i] * 32 + bi[j]] += as[i] * bs[j];
  }
  return g;
}

static CUtensorMap tmap(const void* p, int depth, int rows, int box_rows) {
  CUtensorMap m;
  const cuuint64_t dims[2] = {(cuuint64_t)depth, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)depth};
  const cuuint32_t box[2] = {128, (cuuint32_t)box_rows}, es[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(p), dims, strides, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled: %d\n", int(r)); exit(3); }
  return m;
}

typedef void (*Kern)(const Args);
struct Cfg { int TM, BN, NCT, smem; Kern k[6]; };   // plain, g0, duo32, duo64, duo2_32, duo2_64
static Cfg cfg_p() { return {128, 128, 256, ML<128, 128, 64, 3>::SMEM, {p_plain, p_g0, p_duo32, p_duo64, p_duo2_32, p_duo2_64}}; }
static Cfg cfg_d() { return {128, 32, 256, ML<128, 32, 16, 4>::SMEM, {d_plain, d_g0, d_duo32, d_duo64, d_duo2_32, d_duo2_64}}; }
static const char* GNAME[6] = {"plain", "g0", "duo32", "duo64", "duo2_32", "duo2_64"};
static const int GRS[6] = {0, 0, 32, 64, 32, 64}, GNCH[6] = {0, 0, 4, 4, 8, 8};

struct Lt {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatmulAlgo_t algo;
  void* ws;
  size_t wsz = 64u << 20;
  int algo_id = -1, tile = -1, stages = -1, splitk = -1, ncand = 0;
  float best_ms = 0;
};
// D (m x n row-major) = A (m x k) B^T (B n x k): cuBLASLt's column-major (n, m, k) with its A = our B (op T), B = our A
static void lt_setup(Lt& L, int m, int n, int k, const int8_t* A, const int8_t* B, int* D, cudaStream_t st) {
  CL(cublasLtCreate(&L.h));
  CL(cublasLtMatmulDescCreate(&L.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CL(cublasLtMatrixLayoutCreate(&L.la, CUDA_R_8I, k, n, k));
  CL(cublasLtMatrixLayoutCreate(&L.lb, CUDA_R_8I, k, m, k));
  CL(cublasLtMatrixLayoutCreate(&L.ld, CUDA_R_32I, n, m, n));
  CK(cudaMalloc(&L.ws, L.wsz));
  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &L.wsz, sizeof L.wsz));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  CL(cublasLtMatmulAlgoGetHeuristic(L.h, L.op, L.la, L.lb, L.ld, L.ld, pref, 16, res, &got));
  if (got == 0) { fprintf(stderr, "cuBLASLt: no algorithm\n"); exit(3); }
  L.ncand = got;
  const int32_t one = 1, zero = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float best = 1e30f;
  int bi = -1;
  for (int i = 0; i < got; i++) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    bool ok = true;
    for (int w = 0; w < 2 && ok; w++)
      ok = cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, L.ws, L.wsz, st) ==
           CUBLAS_STATUS_SUCCESS;
    if (!ok) continue;
    CK(cudaEventRecord(e0, st));
    for (int r = 0; r < 5; r++)
      cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, L.ws, L.wsz, st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) { best = ms; bi = i; }
  }
  if (bi < 0) { fprintf(stderr, "cuBLASLt: no algorithm ran\n"); exit(3); }
  L.algo = res[bi].algo;
  L.best_ms = best / 5;
  size_t w;
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_ID, &L.algo_id, sizeof L.algo_id, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &L.tile, sizeof L.tile, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &L.stages, sizeof L.stages, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &L.splitk, sizeof L.splitk, &w);
  CL(cublasLtMatmulPreferenceDestroy(pref));
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
}
static void lt_run(Lt& L, const int8_t* A, const int8_t* B, int* D, cudaStream_t st) {
  const int32_t one = 1, zero = 0;
  CL(cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &L.algo, L.ws, L.wsz, st));
}

// One problem's buffers: NB copies of the weight side (Q(B), B~, P_B) so a decode call reads its weights from HBM, not L2.
struct Prob {
  int m, n, k, nb, grid, group;
  Cfg cfg;
  int8_t *QA, *EA, *XA, *PA;
  std::vector<int8_t*> QB, EB, XB, PB;
  uint32_t *ownA, *invA, *ownB, *invB;   // invA: F_B's column view for side A; invB: F_A's for side B
  int *G, *U, *flag;
  uint32_t* R;
  std::vector<Maps*> maps_plain, maps_noised;   // device copies, per weight copy
  uint32_t seed;
  Lt lt;
  cudaStream_t st;
};

static void fill_codes(std::vector<int8_t>& v, Rng& r, int lo, int hi) {
  for (auto& x : v) x = (int8_t)r.range(lo, hi);
}
template <class T> static T* upload(const std::vector<T>& h) {
  T* d;
  CK(cudaMalloc(&d, h.size() * sizeof(T)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
static Maps* upload_maps(const Maps& m) {
  Maps* d;
  CK(cudaMalloc(&d, sizeof m));
  CK(cudaMemcpy(d, &m, sizeof m, cudaMemcpyHostToDevice));
  return d;
}

struct Host {   // what the gate dumps
  std::vector<int8_t> QA, QB, EA, EB;
  F *fa, *fb;
  std::vector<int> G;
};

static void setup(Prob& P, int m, int n, int k, int grid_cap, uint32_t seed, Host* keep, char force = 0) {
  P.m = m; P.n = n; P.k = k; P.seed = seed;
  P.cfg = force == 'P' ? cfg_p() : force == 'D' ? cfg_d() : m >= 1024 ? cfg_p() : cfg_d();
  if (m % P.cfg.TM || n % P.cfg.BN || k % 128 || k < 256) { fprintf(stderr, "shape %d %d %d not tiled\n", m, n, k); exit(2); }
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const int tiles = (m / P.cfg.TM) * (n / P.cfg.BN);
  P.grid = std::min(tiles, grid_cap > 0 ? grid_cap : nsm);
  P.group = 8;
  const size_t wbytes = (size_t)n * k;
  P.nb = keep ? 1 : (int)std::min<size_t>(8, std::max<size_t>(1, (400ull << 20) / wbytes + 1));
  CK(cudaStreamCreate(&P.st));
  Rng r{seed * 0x100000001B3ull + 17};
  Host h;
  h.QA.resize((size_t)m * k); h.EA.resize((size_t)m * 32);
  fill_codes(h.QA, r, -121, 121); fill_codes(h.EA, r, -3, 3);
  F* fa = new F(k, r);
  F* fb = new F(k, r);
  h.G = gram(*fa, *fb);
  P.QA = upload(h.QA); P.EA = upload(h.EA);
  CK(cudaMalloc(&P.XA, (size_t)m * k)); CK(cudaMalloc(&P.PA, (size_t)m * 128));
  P.ownA = upload(fa->own); P.invA = upload(fb->inv); P.ownB = upload(fb->own); P.invB = upload(fa->inv);
  P.G = upload(h.G);
  for (int i = 0; i < P.nb; i++) {
    h.QB.assign((size_t)n * k, 0); h.EB.assign((size_t)n * 32, 0);
    fill_codes(h.QB, r, -121, 121); fill_codes(h.EB, r, -3, 3);
    P.QB.push_back(upload(h.QB)); P.EB.push_back(upload(h.EB));
    int8_t *xb, *pb;
    CK(cudaMalloc(&xb, wbytes)); CK(cudaMalloc(&pb, (size_t)n * 128));
    P.XB.push_back(xb); P.PB.push_back(pb);
    Maps mp{tmap(P.QA, k, m, P.cfg.TM), tmap(P.QB[i], k, n, P.cfg.BN), tmap(P.PA, 128, m, P.cfg.TM), tmap(pb, 128, n, P.cfg.BN)};
    Maps mn{tmap(P.XA, k, m, P.cfg.TM), tmap(xb, k, n, P.cfg.BN), tmap(P.PA, 128, m, P.cfg.TM), tmap(pb, 128, n, P.cfg.BN)};
    P.maps_plain.push_back(upload_maps(mp)); P.maps_noised.push_back(upload_maps(mn));
  }
  CK(cudaMalloc(&P.U, (size_t)m * n * 4));
  CK(cudaMalloc(&P.R, (size_t)tiles * P.cfg.NCT * 8 * 4));
  CK(cudaMalloc(&P.flag, 4));
  CK(cudaMemset(P.flag, 0, 4));
  std::vector<uint32_t> ka(2 * NPOS), kk(2 * NPOS);
  for (int i = 0; i < 2 * NPOS; i++) { ka[i] = mix32(seed ^ (0xA0000000u + 2 * i)) | 1u; kk[i] = mix32(seed ^ (0xB0000000u + 2 * i + 1)); }
  CK(cudaMemcpyToSymbol(KA, ka.data(), ka.size() * 4));
  CK(cudaMemcpyToSymbol(KK, kk.data(), kk.size() * 4));
  for (int i = 0; i < 6; i++) CK(cudaFuncSetAttribute(P.cfg.k[i], cudaFuncAttributeMaxDynamicSharedMemorySize, P.cfg.smem));
  lt_setup(P.lt, m, n, k, P.QA, P.QB[0], P.U, P.st);
  if (keep) { h.fa = fa; h.fb = fb; *keep = h; } else { delete fa; delete fb; }
}

static void form(Prob& P, int i) {
  form_k<true><<<(P.m + 7) / 8, 256, 0, P.st>>>(P.QA, P.EA, P.ownA, P.invA, P.G, P.XA, P.PA, P.m, P.k, P.flag);
  form_k<false><<<(P.n + 7) / 8, 256, 0, P.st>>>(P.QB[i], P.EB[i], P.ownB, P.invB, nullptr, P.XB[i], P.PB[i], P.n, P.k, P.flag);
}
static void gemm(Prob& P, int kind, int i) {
  Args a{kind == 0 ? P.maps_plain[i] : P.maps_noised[i], P.U, P.R, P.m, P.n, P.k, P.group, P.seed};
  P.cfg.k[kind]<<<P.grid, 384, P.cfg.smem, P.st>>>(a);
}
// the arms: lt (a), plain (my mainloop on Q(A), Q(B)), form, g0 (the noised GEMM and peel alone), and the whole noised call
// form + GEMM: b (no receipts), c (duo, stride 32), d (duo, 64), c2 / d2 (duo2)
static const char* ARMS[] = {"lt", "plain", "form", "g0", "b", "c", "d", "c2", "d2"};
constexpr int NARM = 9;
static void run_arm(Prob& P, int arm, int i) {
  switch (arm) {
    case 0: lt_run(P.lt, P.QA, P.QB[i], P.U, P.st); break;
    case 1: gemm(P, 0, i); break;
    case 2: form(P, i); break;
    case 3: gemm(P, 1, i); break;
    case 4: form(P, i); gemm(P, 1, i); break;
    case 5: form(P, i); gemm(P, 2, i); break;
    case 6: form(P, i); gemm(P, 3, i); break;
    case 7: form(P, i); gemm(P, 4, i); break;
    case 8: form(P, i); gemm(P, 5, i); break;
  }
}

static void dump(const std::string& dir, const char* name, const void* p, size_t bytes, bool dev) {
  std::vector<uint8_t> h(bytes);
  if (dev) CK(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
  else memcpy(h.data(), p, bytes);
  FILE* f = fopen((dir + "/" + name).c_str(), "wb");
  if (!f || fwrite(h.data(), 1, bytes, f) != bytes) { fprintf(stderr, "write %s failed\n", name); exit(3); }
  fclose(f);
}

static int cmd_gate(const std::string& dir, int m, int n, int k, int grid, uint32_t seed, char force) {
  Prob P;
  Host h;
  setup(P, m, n, k, grid, seed, &h, force);
  const int tiles = (m / P.cfg.TM) * (n / P.cfg.BN);
  dump(dir, "QA.bin", h.QA.data(), h.QA.size(), false);
  dump(dir, "QB.bin", h.QB.data(), h.QB.size(), false);
  dump(dir, "EA.bin", h.EA.data(), h.EA.size(), false);
  dump(dir, "EB.bin", h.EB.data(), h.EB.size(), false);
  dump(dir, "ownA.bin", h.fa->own.data(), k * 4, false);
  dump(dir, "ownB.bin", h.fb->own.data(), k * 4, false);
  std::vector<uint32_t> ka(2 * NPOS), kk(2 * NPOS);
  CK(cudaMemcpyFromSymbol(ka.data(), KA, ka.size() * 4));
  CK(cudaMemcpyFromSymbol(kk.data(), KK, kk.size() * 4));
  dump(dir, "KA.bin", ka.data(), ka.size() * 4, false);
  dump(dir, "KK.bin", kk.data(), kk.size() * 4, false);
  // poison every output before each arm, so a stale buffer can't pass
  const size_t ub = (size_t)m * n * 4, rb = (size_t)tiles * P.cfg.NCT * 8 * 4;
  CK(cudaMemset(P.U, 0xA5, ub));
  lt_run(P.lt, P.QA, P.QB[0], P.U, P.st);
  CK(cudaStreamSynchronize(P.st));
  dump(dir, "U_lt.bin", P.U, ub, true);
  CK(cudaMemset(P.XA, 0xA5, (size_t)m * k)); CK(cudaMemset(P.XB[0], 0xA5, (size_t)n * k));
  CK(cudaMemset(P.PA, 0xA5, (size_t)m * 128)); CK(cudaMemset(P.PB[0], 0xA5, (size_t)n * 128));
  form(P, 0);
  CK(cudaStreamSynchronize(P.st));
  dump(dir, "XA.bin", P.XA, (size_t)m * k, true);
  dump(dir, "XB.bin", P.XB[0], (size_t)n * k, true);
  dump(dir, "PA.bin", P.PA, (size_t)m * 128, true);
  dump(dir, "PB.bin", P.PB[0], (size_t)n * 128, true);
  for (int g = 0; g < 6; g++) {
    CK(cudaMemset(P.U, 0xA5, ub)); CK(cudaMemset(P.R, 0xA5, rb));
    gemm(P, g, 0);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    dump(dir, (std::string("U_") + GNAME[g] + ".bin").c_str(), P.U, ub, true);
    if (GRS[g]) dump(dir, (std::string("R_") + GNAME[g] + ".bin").c_str(), P.R, (size_t)tiles * P.cfg.NCT * GNCH[g] * 4, true);
  }
  int flag;
  CK(cudaMemcpy(&flag, P.flag, 4, cudaMemcpyDeviceToHost));
  FILE* f = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(f,
          "{\"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"grid\": %d, \"TM\": %d, \"BN\": %d, \"WM\": %d, \"NCT\": %d, "
          "\"group\": %d, \"limb_overflow\": %d, \"kernels\": [\"plain\", \"g0\", \"duo32\", \"duo64\", \"duo2_32\", \"duo2_64\"]}\n",
          m, n, k, seed, P.grid, P.cfg.TM, P.cfg.BN, P.cfg.BN == 128 ? 64 : 16, P.cfg.NCT, P.group, flag);
  fclose(f);
  printf("gate dump %s: m %d n %d k %d grid %d tiles %d lt algo %d\n", dir.c_str(), m, n, k, P.grid, tiles, P.lt.algo_id);
  return 0;
}

struct Nv {
  nvmlDevice_t d;
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && *u) ? nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS : nvmlDeviceGetHandleByIndex(0, &d) == NVML_SUCCESS;
  }
  unsigned sm() {
    unsigned c = 0;
    if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c);
    return c;
  }
};

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

static int cmd_bench(int reps, const std::vector<std::string>& shapes) {
  Nv nv;
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  for (const auto& sh : shapes) {
    char name[64];
    int m, n, k;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &n, &k) != 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    Prob P;
    setup(P, m, n, k, 0, 0x5EED0000u + m + n + k, nullptr);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int it = 0;
    auto time_arm = [&](int arm, int calls) {
      CK(cudaEventRecord(e0, P.st));
      for (int c = 0; c < calls; c++) run_arm(P, arm, (it++) % P.nb);
      CK(cudaEventRecord(e1, P.st));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      return (double)ms / calls;
    };
    int calls[NARM];
    for (int a = 0; a < NARM; a++) {
      time_arm(a, 2);
      const double t = time_arm(a, 3);
      calls[a] = std::max(2, std::min(400, (int)(3.0 / std::max(t, 1e-4))));
    }
    CK(cudaGetLastError());
    std::vector<double> ms[NARM], mhz[NARM];
    for (int r = 0; r < reps; r++)
      for (int j = 0; j < NARM; j++) {
        const int a = (j + r) % NARM;
        ms[a].push_back(time_arm(a, calls[a]));
        mhz[a].push_back(nv.sm());
      }
    int flag;
    CK(cudaMemcpy(&flag, P.flag, 4, cudaMemcpyDeviceToHost));
    const double macs = (double)m * n * k;
    printf("{\"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"sms\": %d, \"gpu\": \"%s\", \"grid\": %d, \"tile\": \"%dx%d\", "
           "\"weight_copies\": %d, \"limb_overflow\": %d, \"lt\": {\"algo\": %d, \"tile\": %d, \"stages\": %d, \"splitk\": %d, "
           "\"candidates\": %d}, \"arms\": {",
           name, m, n, k, nsm, prop.name, P.grid, P.cfg.TM, P.cfg.BN, P.nb, flag, P.lt.algo_id, P.lt.tile, P.lt.stages,
           P.lt.splitk, P.lt.ncand);
    for (int a = 0; a < NARM; a++) {
      const double t = median(ms[a]), f = median(mhz[a]);
      const double eta = a == 2 ? 0 : macs / (t * 1e-3) / (1024.0 * nsm * f * 1e6);
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"eta\": %.4f, \"calls\": %d, \"ms_all\": [", a ? ", " : "", ARMS[a], t, f,
             eta, calls[a]);
      for (size_t i = 0; i < ms[a].size(); i++) printf("%s%.6f", i ? ", " : "", ms[a][i]);
      printf("], \"mhz_all\": [");
      for (size_t i = 0; i < mhz[a].size(); i++) printf("%s%.0f", i ? ", " : "", mhz[a][i]);
      printf("]}");
    }
    printf("}}\n");
    fflush(stdout);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    CK(cudaDeviceSynchronize());
    CK(cudaDeviceReset());
  }
  return 0;
}

static int cmd_one(const std::string& arm, int m, int n, int k) {
  Prob P;
  setup(P, m, n, k, 0, 0x5EED0000u + m + n + k, nullptr);
  int a = -1;
  for (int i = 0; i < NARM; i++)
    if (arm == ARMS[i]) a = i;
  if (a < 0) { fprintf(stderr, "no arm %s\n", arm.c_str()); return 2; }
  if (a >= 3) form(P, 0);   // the GEMM-only profile reads formed operands
  for (int w = 0; w < 3; w++) run_arm(P, a, 0);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaProfilerStart());
  run_arm(P, a, 0);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 9 && !strcmp(argv[1], "gate"))
    return cmd_gate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), (uint32_t)strtoul(argv[7], 0, 0),
                    argv[8][0]);
  if (argc >= 4 && !strcmp(argv[1], "bench")) {
    std::vector<std::string> s(argv + 3, argv + argc);
    return cmd_bench(atoi(argv[2]), s);
  }
  if (argc >= 6 && !strcmp(argv[1], "one")) return cmd_one(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: fgemm gate DIR M N K GRID SEED P|D | bench REPS name:m:n:k... | one ARM M N K\n");
  return 2;
}
