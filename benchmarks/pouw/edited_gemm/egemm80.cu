// The edited GEMM on sm_80 (A100): egemm.cu's e_l7qs, lpgemm.cu's 16-warp 128 x 128 mainloop with duo
// receipts on C's running sums (each absorb LAG = 7 MMAs after its tile's MMA, a LOP3 tie every 4th MMA) and the peel as
// a 2-atom tail, with sm_80's loads. TMA and mbarrier transactions are sm_90+, so a stage (128 of k for the tile's A and
// B rows) comes in by cp.async, every thread 4 16-byte copies into an NST-stage ring (EG80_NST, 4 by default) laid out
// as TMA's 128-byte swizzle (16-byte chunk c of row r at c ^ (r % 8)), and the CTA meets at one __syncthreads a stage,
// where the stage NST - 1 ahead is issued. The thread layout, the fragments (ldmatrix x4), the saturating IMMA, the receipt chains, keys and finish are
// egemm.cu's (fgemm.cu's duo), so derive.py's tile_receipts checks them over the running sums at each commit boundary.
//   e16  L = 16: IMMA.16816, each atom's two k16 halves as 16 MMAs (every tile's low half, then every tile's high half),
//        an absorb after each, LAG 7 in that order
//   e32  L = 32: IMMA.16832 (e_l7qs)
//   e64  L = 64: IMMA.16832, absorbs on the odd atoms only (egemm.cu's DUO64 at LAG 7, d64_l7q)
//   p32  plain: e32 without receipts or ties (e_plain)
// The peel is IMMA.16832 in every arm (P_A = [S_1 | S_2 | S_0 | 0], P_B = [U | U | U | 0], atoms 0 and 2 at k <= 4096).
// Modes:
//   egemm80 gate DIR M N K SEED       every arm once on 0xA5-poisoned Y and R, Y against the host's X W^T, and the no-write
//                                     control; X, W, U, meta.json and every arm's Y and R to DIR for egemm80_ref.py
//   egemm80 bench REPS M N K          gk::timed_items over every arm: one JSON line an item, then gk's summary line an arm
//   egemm80 hold SECONDS ARM M N K    ARM back to back for SECONDS, one JSON line a second with NVML's SM clock (f_H)
// Build: nvcc -O3 -std=c++17 -arch=sm_80 --no-compress egemm80.cu -ldl
#include "../gpu_kit/bench.cuh"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK GK_CK
#define DEV __device__ __forceinline__

// fgemm.cu's keys, rotations and duo (DUO, lane 0)
__host__ __device__ constexpr uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}
constexpr int NPOS = 16;
__host__ __device__ constexpr uint32_t key_mul(int lane, int t) { return mix32(0xA0000000u + 2u * (lane * NPOS + t)) | 1u; }
__host__ __device__ constexpr uint32_t key_xor(int lane, int t) { return mix32(0xB0000000u + 2u * (lane * NPOS + t) + 1u); }
__host__ __device__ constexpr int rot_of(int t) { return ((t * 7 + 5) % 31) + 1; }
struct St { uint32_t y, z; };
DEV uint32_t rotl(uint32_t x, int r) { return __funnelshift_l(x, x, r); }
template <int T> DEV void duo(St& s, uint32_t p) {
  constexpr uint32_t a = key_mul(0, T), kx = key_xor(0, T);
  const uint32_t u = s.y ^ p ^ kx;
  s.y = u * a + s.z;
  s.z = rotl(u, rot_of(T)) + s.y;
}
DEV uint32_t finish(const St& c) { return c.y ^ rotl(c.z, 16); }
// egemm.cu's tie: d ^= z & zero in one LOP3
DEV void tie(int& d, uint32_t z, uint32_t zero) { asm("lop3.b32 %0, %0, %1, %2, 0x78;" : "+r"(d) : "r"(z), "r"(zero)); }

DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
DEV void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEV void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
DEV void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// d (+)= a b: s8 m16n8k32 (a 4 words a lane, b 2), s8 m16n8k16 (a 2, b 1), fgemm.cu's saturating form
template <bool INIT> DEV void imma32(int (&d)[4], const uint32_t* a, const uint32_t* b) {
  if constexpr (INIT)
    asm("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0));
  else
    asm("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <bool INIT> DEV void imma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  if constexpr (INIT)
    asm("mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %7, %7, %7};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
  else
    asm("mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

struct Args {
  const int8_t* X;    // M x K, the served (dithered) activations
  const int8_t* W;    // N x K, W' = W + U Id^T
  const int8_t* PA;   // M x 128, the tail rows [S_1 | S_2 | S_0 | 0]
  const int8_t* PB;   // N x 128, [U | U | U | 0]
  int* Y;             // M x N, C - S U^T
  uint32_t* R;        // tiles x 512 x 8 receipt words
  int M, N, K;
  uint32_t seed, zero;
};

#ifndef EG80_NST
#define EG80_NST 4  // 3 holds L = 32 and 64 2% and 8% slower; 5 is no faster; L = 16 runs 0.5% faster at 3
#endif
constexpr int BM = 128, BN = 128, BK = 128, NST = EG80_NST, NT = 512, WARPS_N = 4, MI = 2, NI = 4, NU = MI * NI;
static_assert(NST >= 3, "a stage's one barrier frees only the previous step's slot for the refill");
constexpr int A_BYTES = BM * BK, STAGE = A_BYTES + BN * BK, SMEM = NST * STAGE;
enum Arm { P32 = 0, E16 = 1, E32 = 2, E64 = 3 };
constexpr const char* ARM_NAME[] = {"p32", "e16", "e32", "e64"};

template <int ARM> struct G {
  static constexpr bool R = ARM != P32, K16 = ARM == E16, HALF = ARM == E64;
  // atom POS of a stage commits (E64: the odd ones, so the odd k32 steps)
  __host__ __device__ static constexpr bool rc(int pos) { return R && (!HALF || pos % 2 == 1); }

  uint32_t sbase, a_off[4], b_off[4];
  const Args& a;
  int tiles_m, tiles_n, KT, ptile, pkt, ps;
  // the copier's state: this thread's 4 copies a step are rows r0 and r0 + 64 of the A tile (X) and of the B tile (W) at
  // byte offsets o[0..3] (PA, PB at the tail step), into smem at d0, d0 + 64 BK and A_BYTES past both; ta, tb: the tail's
  uint32_t o[4], ta, tb, d0;

  DEV G(const Args& args, uint8_t* smem) : a(args) {
    sbase = smem_u32(smem);
    tiles_m = a.M / BM, tiles_n = a.N / BN, KT = a.K / BK;
    ptile = blockIdx.x, pkt = 0, ps = 0;
    const int r0 = threadIdx.x >> 3, ch = threadIdx.x & 7;
    d0 = r0 * BK + ((ch ^ (r0 & 7)) << 4);
    retarget();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N, x = lane & 7;
    const int a_row = wm * 32 + (lane & 15), a_half = lane >> 4;
    const int b_row = wn * 32 + (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      a_off[kk] = a_row * BK + (((2 * kk + a_half) ^ x) << 4);
      b_off[kk] = A_BYTES + b_row * BK + (((2 * kk + b_half) ^ x) << 4);
    }
  }
  // lpgemm's raster: group 8 of tile rows (tiles_m % 8 == 0)
  DEV void place(int t, int& tm, int& tn) const {
    const int g = 8 * tiles_n, blk = t / g, r = t - blk * g;
    tm = 8 * blk + (r & 7), tn = r >> 3;
  }
  // the copier's offsets at stage 0 of tile ptile (the one division a tile)
  DEV void retarget() {
    if (ptile >= tiles_m * tiles_n) return;
    int tm, tn;
    place(ptile, tm, tn);
    const uint32_t r0 = threadIdx.x >> 3, c16 = (threadIdx.x & 7) * 16;
    o[0] = (tm * BM + r0) * (uint32_t)a.K + c16, o[1] = o[0] + 64u * a.K;
    o[2] = (tn * BN + r0) * (uint32_t)a.K + c16, o[3] = o[2] + 64u * a.K;
    ta = (tm * BM + r0) * BK + c16, tb = (tn * BN + r0) * BK + c16;
  }
  // the next step's copies (tile ptile, stage pkt; pkt == KT: the tail rows) into slot ps, then one commit group
  DEV void issue() {
    if (ptile < tiles_m * tiles_n) {
      const bool tail = pkt == KT;
      const int8_t *pa = tail ? a.PA : a.X, *pb = tail ? a.PB : a.W;
      const uint32_t dst = sbase + ps * STAGE + d0;
      cp16(dst, pa + o[0]);
      cp16(dst + 64 * BK, pa + o[1]);
      cp16(dst + A_BYTES, pb + o[2]);
      cp16(dst + A_BYTES + 64 * BK, pb + o[3]);
      if (++pkt < KT) {
#pragma unroll
        for (int i = 0; i < 4; i++) o[i] += BK;
      } else if (pkt == KT) {
        o[0] = ta, o[1] = ta + 64 * BK, o[2] = tb, o[3] = tb + 64 * BK;
      } else {
        pkt = 0, ptile += gridDim.x;
        retarget();
      }
    }
    cp_commit();
    if (++ps == NST) ps = 0;
  }
  // wait for this step's stage (slot s); every warp is past the previous step, so its slot takes the step NST - 1 ahead
  DEV uint32_t next_stage(int& s) {
    cp_wait<NST - 2>();
    __syncthreads();
    issue();
    const uint32_t sb = sbase + s * STAGE;
    if (++s == NST) s = 0;
    return sb;
  }
  DEV void load(uint32_t (&fa)[MI][4], uint32_t (&fb)[NI][2], uint32_t sb, int kk) const {
#pragma unroll
    for (int mi = 0; mi < MI; mi++) ldsm4(fa[mi], sb + a_off[kk] + mi * 16 * BK);
#pragma unroll
    for (int nj = 0; nj < NI / 2; nj++) {
      uint32_t r[4];
      ldsm4(r, sb + b_off[kk] + nj * 16 * BK);
      fb[2 * nj][0] = r[0], fb[2 * nj][1] = r[1], fb[2 * nj + 1][0] = r[2], fb[2 * nj + 1][1] = r[3];
    }
  }
  // unit J (tile mi NI + ni): its four words into chains 4 (J % 2) + e at key position J (fgemm.cu's absorb)
  template <int J> DEV static void ab(St (&ch)[8], const int (&acc)[MI][NI][4]) {
    constexpr int H = 4 * (J % 2);
#pragma unroll
    for (int e = 0; e < 4; e++) duo<J>(ch[H + e], (uint32_t)acc[J / NI][J % NI][e]);
  }
  // one pass of the 8 tiles' MMAs (a k32 atom, or E16's low or high k16 half of one): after MMA IDX, the previous pass's
  // unit IDX + 1 and this pass's unit IDX - 7 (LAG 7); FIRST: the tile's first pass (no previous pass, C = 0)
  template <bool FIRST, bool PREV, bool SELF, int H, int IDX = 0>
  DEV static void pass(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&fa)[MI][4], const uint32_t (&fb)[NI][2],
                       uint32_t zero) {
    if constexpr (IDX < NU) {
      constexpr int mi = IDX / NI, ni = IDX % NI;
      if constexpr (R && !FIRST && IDX % 4 == 0) tie(acc[mi][ni][0], ch[3].z, zero);
      if constexpr (K16) imma16<FIRST>(acc[mi][ni], fa[mi][2 * H], fa[mi][2 * H + 1], fb[ni][H]);
      else imma32<FIRST>(acc[mi][ni], fa[mi], fb[ni]);
      if constexpr (PREV && !FIRST && IDX + 1 < NU) ab<IDX + 1>(ch, acc);
      if constexpr (SELF && IDX == NU - 1) ab<0>(ch, acc);
      pass<FIRST, PREV, SELF, H, IDX + 1>(acc, ch, fa, fb, zero);
    }
  }
  // atom POS of a stage: E16's two passes, else one
  template <bool FIRST, int POS>
  DEV static void atom(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&fa)[MI][4], const uint32_t (&fb)[NI][2],
                       uint32_t zero) {
    if constexpr (K16) {
      pass<FIRST, R, R, 0>(acc, ch, fa, fb, zero);
      pass<false, R, R, 1>(acc, ch, fa, fb, zero);
    } else {
      pass<FIRST, rc((POS + 3) % 4), rc(POS), 0>(acc, ch, fa, fb, zero);
    }
  }
  template <bool FIRST> DEV void stage(int& s, int (&acc)[MI][NI][4], St (&ch)[8], uint32_t zero) {
    const uint32_t sb = next_stage(s);
    uint32_t fa[2][MI][4], fb[2][NI][2];
    load(fa[0], fb[0], sb, 0);
    load(fa[1], fb[1], sb, 1);
    atom<FIRST, 0>(acc, ch, fa[0], fb[0], zero);
    load(fa[0], fb[0], sb, 2);
    atom<false, 1>(acc, ch, fa[1], fb[1], zero);
    load(fa[1], fb[1], sb, 3);
    atom<false, 2>(acc, ch, fa[0], fb[0], zero);
    atom<false, 3>(acc, ch, fa[1], fb[1], zero);
  }
  // the last pass's units 1..7, after the tile's last MMA
  template <int J = 1> DEV static void flush(St (&ch)[8], const int (&acc)[MI][NI][4]) {
    if constexpr (R && J < NU) {
      ab<J>(ch, acc);
      flush<J + 1>(ch, acc);
    }
  }
  // the tail stage: acc -= 256 S_1 U^T + S_0 U^T (atoms 0 and 2 of P_A, P_B)
  DEV void peel(int& s, int (&acc)[MI][NI][4]) {
    const uint32_t sb = next_stage(s);
    uint32_t fb[2][NI][2], fa[2][MI][4];
    load(fa[0], fb[0], sb, 0);
    load(fa[1], fb[1], sb, 2);
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int t1[4], t2[4];
        imma32<true>(t1, fa[0][mi], fb[0][ni]);
        imma32<true>(t2, fa[1][mi], fb[1][ni]);
#pragma unroll
        for (int e = 0; e < 4; e++)
          acc[mi][ni][e] = (int)((uint32_t)acc[mi][ni][e] - ((uint32_t)t1[e] << 8) - (uint32_t)t2[e]);
      }
  }
};

template <int ARM> __global__ void __launch_bounds__(NT, 1) egemm80(const __grid_constant__ Args a) {
  using L = G<ARM>;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(a, smem);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, ctid = threadIdx.x;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N, g = lane >> 2, q = lane & 3;
  for (int i = 0; i < NST - 1; i++) ml.issue();
  int s = 0;
  const int tiles = ml.tiles_m * ml.tiles_n;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    int tm, tn;
    ml.place(t, tm, tn);
    const uint32_t tile_id = tm * ml.tiles_n + tn;
    St ch[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      ch[e].y = L::R ? mix32(a.seed ^ mix32((tile_id * NT + ctid) * 8 + e)) : 0u;
      ch[e].z = L::R ? mix32(ch[e].y + 0x9E3779B9u) : 0u;
    }
    int acc[MI][NI][4];
    ml.template stage<true>(s, acc, ch, a.zero);
#pragma unroll 1
    for (int kt = 1; kt < ml.KT; kt++) ml.template stage<false>(s, acc, ch, a.zero);
    L::flush(ch, acc);
    ml.peel(s, acc);
    const int row0 = tm * BM + wm * 32 + g, col0 = tn * BN + wn * 32 + 2 * q;
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int* p = a.Y + (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni;
        *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
        *reinterpret_cast<int2*>(p + (size_t)8 * a.N) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
      }
    if constexpr (L::R) {
      uint4* r = reinterpret_cast<uint4*>(a.R) + ((size_t)tile_id * NT + ctid) * 2;
      r[0] = make_uint4(finish(ch[0]), finish(ch[1]), finish(ch[2]), finish(ch[3]));
      r[1] = make_uint4(finish(ch[4]), finish(ch[5]), finish(ch[6]), finish(ch[7]));
    }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
}

// ---------------------------------------------------------------- host
using Kern = void (*)(Args);
static const Kern KERNS[] = {egemm80<P32>, egemm80<E16>, egemm80<E32>, egemm80<E64>};
static int g_sms = 0;

struct Bufs {
  int M, N, K;
  int8_t *X, *W, *PA, *PB;
  int* Y;
  uint32_t* R;
  std::vector<int8_t> hX, hW, hU;   // X, W (before U) and U (N x 32)
  size_t ybytes() const { return (size_t)M * N * 4; }
  size_t rbytes() const { return (size_t)(M / BM) * (N / BN) * NT * 8 * 4; }
};

// X uniform on [-127, 127] (the dithered range), W on [-63, 63], U on ±1..±64; W' = W + U Id^T, S = X Id in limbs
static Bufs make(int M, int N, int K, uint32_t seed) {
  if (M % (8 * BM) || N % BN || K % BK || K > 4096) {
    fprintf(stderr, "ERR shape: M %% 1024, N %% 128, K %% 128 == 0 and K <= 4096 (the 2-atom tail)\n");
    exit(2);
  }
  Bufs b{M, N, K};
  std::mt19937 rng(seed);
  b.hX.resize((size_t)M * K), b.hW.resize((size_t)N * K), b.hU.resize((size_t)N * 32);
  for (auto& v : b.hX) v = (int8_t)((int)(rng() % 255) - 127);
  for (auto& v : b.hW) v = (int8_t)((int)(rng() % 127) - 63);
  for (auto& v : b.hU) { const int m = (int)(rng() % 64) + 1; v = (int8_t)((rng() & 1) ? -m : m); }
  std::vector<int8_t> Wp((size_t)N * K), PA((size_t)M * BK, 0), PB((size_t)N * BK, 0);
  for (int j = 0; j < N; j++)
    for (int k = 0; k < K; k++) Wp[(size_t)j * K + k] = (int8_t)(b.hW[(size_t)j * K + k] + b.hU[(size_t)j * 32 + k % 32]);
  for (int i = 0; i < M; i++)
    for (int c = 0; c < 32; c++) {
      int S = 0;
      for (int t = 0; t < K / 32; t++) S += b.hX[(size_t)i * K + 32 * t + c];
      const int s0 = (int8_t)(S & 0xFF), s1 = std::max(-127, std::min(127, (S - s0) / 256));
      if (256 * s1 + s0 != S) { fprintf(stderr, "ERR S_2 != 0 at k = %d\n", K); exit(2); }
      PA[(size_t)i * BK + c] = (int8_t)s1, PA[(size_t)i * BK + 64 + c] = (int8_t)s0;
    }
  for (int j = 0; j < N; j++)
    for (int c = 0; c < 32; c++)
      for (int r = 0; r < 3; r++) PB[(size_t)j * BK + 32 * r + c] = b.hU[(size_t)j * 32 + c];
  CK(cudaMalloc(&b.X, b.hX.size()));
  CK(cudaMalloc(&b.W, Wp.size()));
  CK(cudaMalloc(&b.PA, PA.size()));
  CK(cudaMalloc(&b.PB, PB.size()));
  CK(cudaMalloc(&b.Y, b.ybytes()));
  CK(cudaMalloc(&b.R, b.rbytes()));
  CK(cudaMemcpy(b.X, b.hX.data(), b.hX.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b.W, Wp.data(), Wp.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b.PA, PA.data(), PA.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b.PB, PB.data(), PB.size(), cudaMemcpyHostToDevice));
  return b;
}

static void launch(int arm, const Bufs& b, uint32_t seed, cudaStream_t s) {
  Args a{b.X, b.W, b.PA, b.PB, b.Y, b.R, b.M, b.N, b.K, seed, 0u};
  const int tiles = (b.M / BM) * (b.N / BN);
  KERNS[arm]<<<std::min(g_sms, tiles), NT, SMEM, s>>>(a);
}

static double mmas(int arm, const Bufs& b) {   // warp MMAs a call, the tail's included
  const double tiles = (double)(b.M / BM) * (b.N / BN), main = tiles * 16 * NU * (b.K / (arm == E16 ? 16 : 32));
  return main + tiles * 16 * NU * 2;
}

static int gate(const char* dir, int M, int N, int K, uint32_t seed, cudaStream_t s) {
  Bufs b = make(M, N, K, seed);
  std::vector<int> hY((size_t)M * N);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      int acc = 0;
      for (int k = 0; k < K; k++) acc += (int)b.hX[(size_t)i * K + k] * b.hW[(size_t)j * K + k];
      hY[(size_t)i * N + j] = acc;
    }
  gk::Dump d(dir);
  d.bytes("X", b.X, b.hX.size(), s);
  FILE* f = fopen((std::string(dir) + "/W.bin").c_str(), "wb");
  fwrite(b.hW.data(), 1, b.hW.size(), f);
  fclose(f);
  f = fopen((std::string(dir) + "/U.bin").c_str(), "wb");
  fwrite(b.hU.data(), 1, b.hU.size(), f);
  fclose(f);
  int bad = 0;
  std::string arms;
  for (int arm = 0; arm < 4; arm++) {
    auto run = [&](cudaStream_t st) { launch(arm, b, seed, st); CK(cudaGetLastError()); };
    const gk::PoisonGate gy = gk::poison_gate(b.Y, b.ybytes(), s, run, [&](const void* h, size_t n) {
      return std::memcmp(h, hY.data(), n) == 0;
    });
    gk::PoisonGate gr;
    if (arm != P32)
      gr = gk::poison_gate(b.R, b.rbytes(), s, run, [&](const void* h, size_t n) {
        return gk::poisoned_words(h, n) == 0;
      });
    // each gate's control leaves its buffer poisoned: one more launch, so the dump holds the arm's own output
    CK(cudaMemsetAsync(b.Y, gk::POISON, b.ybytes(), s));
    CK(cudaMemsetAsync(b.R, gk::POISON, b.rbytes(), s));
    run(s);
    CK(cudaStreamSynchronize(s));
    if (arm != P32) d.bytes(std::string("R_") + ARM_NAME[arm], b.R, b.rbytes(), s);
    d.bytes(std::string("Y_") + ARM_NAME[arm], b.Y, b.ybytes(), s);
    const bool ok = gy.ok() && (arm == P32 || gr.ok());
    bad += !ok;
    printf("{\"gate\":\"%s\",\"y_equal\":%d,\"y_control_rejected\":%d,\"y_left\":%zu,\"r_unpoisoned\":%d,"
           "\"r_control_rejected\":%d,\"r_left\":%zu,\"ok\":%d}\n",
           ARM_NAME[arm], gy.accepted, gy.control_rejected, gy.left, arm == P32 ? -1 : (int)gr.accepted,
           arm == P32 ? -1 : (int)gr.control_rejected, gr.left, ok);
    arms += std::string(arms.empty() ? "" : ",") + "\"" + ARM_NAME[arm] + "\"";
  }
  f = fopen((std::string(dir) + "/meta.json").c_str(), "w");
  fprintf(f, "{\"M\":%d,\"N\":%d,\"K\":%d,\"seed\":%u,\"arms\":[%s]}\n", M, N, K, seed, arms.c_str());
  fclose(f);
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: egemm80 gate DIR M N K SEED | bench REPS M N K | hold SECONDS ARM M N K\n");
    return 2;
  }
  const std::string mode = argv[1];
  CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0));
  for (Kern k : KERNS) CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  gk::Nvml nv(0);
  if (mode == "gate" && argc == 7) return gate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)atol(argv[6]), s);
  if (mode == "bench" && argc == 6) {
    const int reps = atoi(argv[2]);
    Bufs b = make(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), 1);
    std::vector<gk::Arm> arms;
    for (int arm = 0; arm < 4; arm++)
      arms.push_back({ARM_NAME[arm], [&, arm](cudaStream_t st, int n) {
                        for (int i = 0; i < n; i++) launch(arm, b, 7u, st);
                      }});
    const gk::Timing t = gk::timed_items(arms, reps, s, nv);
    gk::items_jsonl(t, arms, stdout);
    for (int arm = 0; arm < 4; arm++)
      printf("{\"arm\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"sms\":%d,\"mmas\":%.0f,\"warmup_ms\":%.1f,\"steady\":%d}\n",
             ARM_NAME[arm], b.M, b.N, b.K, g_sms, mmas(arm, b), t.warmup_ms, (int)t.steady);
    gk::summary(t, arms);
    return 0;
  }
  if (mode == "hold" && argc == 7) {
    const double secs = atof(argv[2]);
    int arm = -1;
    for (int i = 0; i < 4; i++) if (std::string(argv[3]) == ARM_NAME[i]) arm = i;
    if (arm < 0) { fprintf(stderr, "ERR arm %s\n", argv[3]); return 2; }
    Bufs b = make(atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), 1);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const auto t0 = std::chrono::steady_clock::now();
    for (int sec = 0;; sec++) {
      int calls = 0;
      CK(cudaEventRecord(e0, s));
      const auto c0 = std::chrono::steady_clock::now();
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count() < 1.0) {
        for (int i = 0; i < 8; i++) launch(arm, b, 7u, s);
        calls += 8;
        CK(cudaStreamSynchronize(s));
      }
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      const gk::Sample smp = nv.sample();
      printf("{\"hold\":\"%s\",\"sec\":%d,\"calls\":%d,\"ms_per_call\":%.6f,\"sm_mhz\":%u,\"throttle\":\"0x%llx\",\"mmas\":%.0f,"
             "\"sms\":%d}\n",
             ARM_NAME[arm], sec, calls, ms / calls, smp.sm_mhz, smp.throttle, mmas(arm, b), g_sms);
      fflush(stdout);
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= secs) break;
    }
    return 0;
  }
  fprintf(stderr, "usage: egemm80 gate DIR M N K SEED | bench REPS M N K | hold SECONDS ARM M N K\n");
  return 2;
}
