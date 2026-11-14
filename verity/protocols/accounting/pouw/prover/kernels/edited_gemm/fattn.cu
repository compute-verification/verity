// H1's attention on sm_120a, S never stored: cohalf's QK^T (cgemm.cu's h_l7qs) with its words and receipts,
// the softmax, and the adopted P.V's P~ path (apv.cu's apv_fu), for Llama-3.1-8B's GQA (H query heads, kvh = H / rep KV
// heads, head dimension 128, s tokens, causal).
//
// Operands, one KV head's block at a time: keys x (bf16, kvh s x 128) through hquant (X = clamp(floor(x / 2), -64, 62) + b,
// P_A = [Y_lo | Y_hi | 0 | 0]), queries Q (int8 in [-64, 62], H s x 128, head h = kvh rep + j at rows h s ..) through
// cgemm.cu's hwedit<true> (W' = Q + c + U, P_B = [-U_lo | -U_hi | 0 | 0], and W^ = Q + c, the served queries). For KV head
// kvh, cohalf's O = X_kvh W'^T's tail-corrected value is S^T: rows the keys, columns its rep s queries (query column col
// at position pos = col mod s).
//
// The softmax (this file's reference, `fattn_ref.py` on the CPU): for query (h, pos), S(key) for keys 0 .. pos, m its max,
// e(key) = LUT[min(umulhi(m - S(key), A), 255)] with LUT[t] = rint(127 2^(-t / 16)) (so the max key's e is 127), P_prob =
// e at keys 1 .. pos (0 at key 0 and past pos, as apv.cu's P_prob), L = sum of e over keys 0 .. pos, s_q = 1 / L and
// z_q = e(0) / L (IEEE float division), apv.cu's s_q and z_q: its f = (float(O + 64 S) s_q + z_q float(X(0))) s_ch.
//
// Kernels:
//   hm_l7qs   pass 1: cohalf's tiles from a table (KV head, key tile tm, query tile tn), the causal ones (tm <= tn mod
//             s / 128) or all: h_l7qs's mainloop, tail and duo receipts op for op (tile tm tiles_n + tn, seed
//             fa_seed(seed, kvh)), and in place of O's store each query column's max over its keys <= pos (atomicMax)
//   hsm       pass 2: S recomputed as a plain GEMM W^ X^T on the same table (rows the queries), the softmax in its
//             epilogue: P_prob's bytes to P, e(0) to E0, the row's sum of e (atomicAdd) to L
//   lred      s_q and z_q from L and E0
//   hsref     the unfused reference: h_l7qs's stored O (one KV head), a thread a query column: P_prob, s_q, z_q
// The fused attention is hquant, hwedit<true>, hm_l7qs, hsm, lred, then apv.cu's uedit, vedit, vsum and apv_fu on P_prob.
// Modes:
//   fattn gate DIR S H REP SEED   poisoned buffers. Unfused: h_l7qs a KV head at a time (S stored), hsref, apv.cu's four
//                                 producers and apv_l7qs. Fused: hm_l7qs (causal; and over all tiles), hsm, lred, uedit,
//                                 vedit, vsum, apv_fu. Bitwise: hm's receipts against h_l7qs's on every tile it runs, P_prob
//                                 on every byte apv reads, s_q, z_q, apv_fu's f and receipts against apv_l7qs's (and again
//                                 with vfold for vedit and vsum, from vfold's outputs poisoned); a no-write control; sampled
//                                 query columns' X, W^, S, P_prob, s_q, z_q to DIR for fattn_ref.py
//   fattn bench REPS S H REP      hot items interleaved across FA_ARMS, one JSON line
//   fattn one ARM S H REP         three warm-up calls, then one between cudaProfilerStart and Stop (for ncu)
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a fattn.cu -lcuda -lcublasLt -lnvidia-ml
// Run: fattnrun.sh
#define APV_LIB
#include "apv.cu"
#include <climits>
#include <cmath>
#include <type_traits>

constexpr int FA_LUT = 256;
__device__ uint8_t fa_lut[FA_LUT];
__host__ __device__ inline uint32_t fa_seed(uint32_t seed, int kvh) { return seed + 0x9E3779B9u * (uint32_t)kvh; }

struct FaArgs {
  const Maps* maps;   // hm: a X, b W', pa P_A, pb P_B; hsm: a W^, b X
  const int4* tab;    // a tile: tm (key tile), tn (query tile of the KV head), kvh
  int ntiles, s, rep;
  uint32_t seed, A;
  uint32_t* R;
  int *M, *L, *E0;
  int8_t* P;
};

// EL's loop on one-stage tiles (K = 128) from a table, as apv.cu's APL: R, cohalf's (the tail stage, keys on A); else
// the plain recompute (queries on A, no tail)
template <bool R> struct FL : EL<R ? DUO : NONE, R ? 7 : 0, R ? 4 : 0, true> {
  using E = EL<R ? DUO : NONE, R ? 7 : 0, R ? 4 : 0, true>;
  using E::s; using E::pt; using E::pkt; using E::ps; using E::pph; using E::maps; using E::issuer;
  using E::stage; using E::full; using E::empty; using E::advance; using E::wait_full; using E::load;
  static constexpr int MI = E::MI, NI = E::NI, NKR = E::NKR;
  const int4* tab = nullptr;
  int ntiles = 0, s_len = 0, rs_len = 0;
  int jpt = -1, jA = 0, jB = 0;
  DEV FL(uint8_t* smem, const Maps* m) : E(smem, m, 128, 128, 128, 8, R, 1) {}
  DEV void issue() {
    if (pt >= ntiles) return;
    if (pt != jpt) {
      const int4 e = __ldg(tab + pt);
      const int kr = e.z * s_len + 128 * e.x, qr = e.z * rs_len + 128 * e.y;
      jA = R ? kr : qr; jB = R ? qr : kr;
      jpt = pt;
    }
    mbar_wait(empty(ps), pph ^ 1);
    mbar_expect_tx(full(ps), E::STAGE);
    if (pkt == 0) {
      tma_load(stage(ps), &maps->a, full(ps), 0, jA);
      tma_load(stage(ps) + E::A_BYTES, &maps->b, full(ps), 0, jB);
    } else {
      tma_load(stage(ps), &maps->pa, full(ps), 0, jA);
      tma_load(stage(ps) + E::A_BYTES, &maps->pb, full(ps), 0, jB);
    }
    if (++pkt == (R ? 2 : 1)) { pkt = 0; pt += gridDim.x; }
    if (++ps == E::NST) { ps = 0; pph ^= 1; }
  }
  DEV void prologue() {
    if (issuer) {
      tma_acquire(&maps->a); tma_acquire(&maps->b);
      if (R) { tma_acquire(&maps->pa); tma_acquire(&maps->pb); }
      for (int i = 0; i < E::NST; i++) issue();
    }
    __syncwarp();
  }
  DEV void release(uint32_t done) {
    mbar_release(done);
    if (issuer) issue();
    __syncwarp();
  }
  // EL's tile_il at KT = 1 (stage_il<true, true>, then the flush), on this release
  DEV void tile1(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[NKR], uint32_t zero) {
    uint32_t fa[2][MI][4], fb[2][NI][2];
    wait_full();
    const uint32_t sb = stage(s), done = empty(s);
    load(fa[0], fb[0], sb, 0);
    load(fa[1], fb[1], sb, 1);
    E::template atom<true, 0>(acc, ch, kr, fa[0], fb[0], zero);
    load(fa[0], fb[0], sb, 2);
    E::template atom<false, 1>(acc, ch, kr, fa[1], fb[1], zero);
    load(fa[1], fb[1], sb, 3);
    E::template atom<false, 2>(acc, ch, kr, fa[0], fb[0], zero);
    advance();
    E::template atom<false, 3>(acc, ch, kr, fa[1], fb[1], zero);
    release(done);
    if constexpr (R) E::flush(ch, kr, acc);
  }
};

template <bool H1> DEV void hm_body(const FaArgs& a) {
  using L = FL<true>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps);
  ml.tab = a.tab; ml.ntiles = a.ntiles; ml.s_len = a.s; ml.rs_len = a.rep * a.s;
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  const int tiles_n = a.rep * a.s / 128;
  const size_t tpk = (size_t)(a.s / 128) * tiles_n;
  uint32_t kr[L::NKR];
  kr[0] = 0;
  for (int t = blockIdx.x; t < a.ntiles; t += gridDim.x) {
    const int4 te = __ldg(a.tab + t);
    const int tm = te.x, tn = te.y, kvh = te.z;
    const uint32_t tile_id = tm * tiles_n + tn, sd = fa_seed(a.seed, kvh);
    St ch[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      ch[e].y = mix32(sd ^ mix32((tile_id * L::NCT + ctid) * 8 + e));
      ch[e].z = mix32(ch[e].y + 0x9E3779B9u);
    }
    int acc[MI][NI][4];
    ml.tile1(acc, ch, kr, 0u);
    if constexpr (H1) htail_h1<L>(ml, acc, ch);
    else htail<L>(ml, acc, ch);
    const int row0 = tm * 128 + wm * 32 + g, col0 = tn * 128 + wn * 32 + 2 * q;
    // any-s (s a multiple of 256, not only a power of 2): a query column's position is col mod s, and the column's rep
    // block is 128-aligned, so pos = (tn mod tiles per block) 128 + the column's offset in the tile. The table carries
    // that (tn mod tiles per block) in .w (host-side), so the device does no modulo (no per-column & (s - 1), which only
    // holds for a power-of-2 s, and no runtime integer division, which the SASS gate's int seed doesn't always match)
    const int tnm = te.w, posb = tnm * 128 + wn * 32 + 2 * q;
    int* mc = a.M + (size_t)kvh * a.rep * a.s + col0;
    // a tile strictly below the diagonal needs no key <= pos test (EDGE: the tiles that do)
    auto colmax = [&](auto edge_t) {
      constexpr bool EDGE = decltype(edge_t)::value;
#pragma unroll
      for (int ni = 0; ni < NI; ni++)
#pragma unroll
        for (int c = 0; c < 2; c++) {
          const int pos = posb + 8 * ni + c;
          int mx = INT_MIN;
#pragma unroll
          for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int h = 0; h < 2; h++)
              if (!EDGE || row0 + 16 * mi + 8 * h <= pos) mx = max(mx, acc[mi][ni][2 * h + c]);
          mx = max(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
          mx = max(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
          mx = max(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 16));
          if (g == 0 && mx != INT_MIN) atomicMax(mc + 8 * ni + c, mx);
        }
    };
    if (tm >= tnm) colmax(std::true_type{});
    else colmax(std::false_type{});
    uint4* r = reinterpret_cast<uint4*>(a.R) + (((size_t)kvh * tpk + tile_id) * L::NCT + ctid) * 2;
    r[0] = make_uint4(finish<DUO>(ch, 0), finish<DUO>(ch, 1), finish<DUO>(ch, 2), finish<DUO>(ch, 3));
    r[1] = make_uint4(finish<DUO>(ch, 4), finish<DUO>(ch, 5), finish<DUO>(ch, 6), finish<DUO>(ch, 7));
  }
}
__global__ void __launch_bounds__(MLW::NT, 1) hm_l7qs(const __grid_constant__ FaArgs a) { hm_body<false>(a); }
// cohalf H1 (co32c-u16): the digit-split tail (htail_h1), Q/K floored at top 111
__global__ void __launch_bounds__(MLW::NT, 1) hm_l7qs_h1(const __grid_constant__ FaArgs a) { hm_body<true>(a); }

__global__ void __launch_bounds__(MLW::NT, 1) hsm(const __grid_constant__ FaArgs a) {
  using L = FL<false>;
  constexpr int MI = L::MI, NI = L::NI;
  __shared__ uint8_t lut[FA_LUT];
  extern __shared__ __align__(1024) uint8_t smem[];
  for (int i = threadIdx.x; i < FA_LUT; i += blockDim.x) lut[i] = fa_lut[i];
  L ml(smem, a.maps);
  ml.tab = a.tab; ml.ntiles = a.ntiles; ml.s_len = a.s; ml.rs_len = a.rep * a.s;
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3;
  uint32_t kr[L::NKR];
  kr[0] = 0;
  St ch[8];
  const bool b1 = q & 2, b0 = q & 1;
  for (int t = blockIdx.x; t < a.ntiles; t += gridDim.x) {
    const int4 te = __ldg(a.tab + t);
    const int tm = te.x, tn = te.y, kvh = te.z;
    const int qr0 = tn * 128 + wm * 32 + g, k0 = tm * 128 + wn * 32 + 2 * q;
    const int tnm = te.w, posb = tnm * 128 + wm * 32 + g;   // any-s: the query tile's rep-block position from the table (.w), so the device needs no modulo (SASS-clean)
    const size_t qb = (size_t)kvh * a.rep * a.s;
    int m[MI][2];
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int h = 0; h < 2; h++) m[mi][h] = __ldg(a.M + qb + qr0 + 16 * mi + 8 * h);
    int acc[MI][NI][4];
    ml.tile1(acc, ch, kr, 0u);
    // a tile strictly below the diagonal and off key 0 needs neither test (EDGE: the tiles that do)
    auto epi = [&](auto edge_t) {
      constexpr bool EDGE = decltype(edge_t)::value;
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int qr = qr0 + 16 * mi + 8 * h, pos = posb + 16 * mi + 8 * h;
          int sum = 0;
          uint32_t pw[NI];
#pragma unroll
          for (int ni = 0; ni < NI; ni++) {
            pw[ni] = 0;
#pragma unroll
            for (int c = 0; c < 2; c++) {
              const int key = k0 + 8 * ni + c;
              const uint32_t tt = min(__umulhi((uint32_t)(m[mi][h] - acc[mi][ni][2 * h + c]), a.A), (uint32_t)(FA_LUT - 1));
              int e = lut[tt];
              if (EDGE && key > pos) e = 0;
              sum += e;
              if (EDGE && key == 0) a.E0[qb + qr] = e;
              else pw[ni] |= (uint32_t)e << (8 * c);
            }
          }
          // the quad's 4 x 4 16-bit pieces transposed so lane q stores keys 8q .. 8q + 7 of the warp's 32 in one 8-byte store
          const uint32_t A01 = pw[0] | pw[1] << 16, A23 = pw[2] | pw[3] << 16;
          const uint32_t X = b1 ? A23 : A01, Y = __shfl_xor_sync(0xFFFFFFFFu, b1 ? A01 : A23, 2);
          const uint32_t r2 = __shfl_xor_sync(0xFFFFFFFFu, b0 ? ((X & 0xFFFFu) | (Y << 16)) : ((X >> 16) | (Y & 0xFFFF0000u)), 1);
          const uint32_t mx = b0 ? X >> 16 : X & 0xFFFFu, my = b0 ? Y >> 16 : Y & 0xFFFFu, rlo = r2 & 0xFFFFu, rhi = r2 >> 16;
          const uint32_t P1 = b0 ? (rlo | mx << 16) : (mx | rlo << 16), P2 = b0 ? (rhi | my << 16) : (my | rhi << 16);
          *reinterpret_cast<uint2*>(a.P + (qb + qr) * a.s + tm * 128 + wn * 32 + 8 * q) = make_uint2(b1 ? P2 : P1, b1 ? P1 : P2);
          sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1);
          sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2);
          if (q == 0) atomicAdd(a.L + qb + qr, sum);
        }
    };
    if (tm == 0 || tm >= tnm) epi(std::true_type{});
    else epi(std::false_type{});
  }
}

// float32's IEEE quotient a / L through double: with 0 <= a <= 127 and 1 <= L < 2^24, a / L is a float or at least 2^-31
// of itself from a float midpoint, so double's rounding can't move float's (FP32 div.rn shares one reciprocal between
// the two quotients of a divisor, which the SASS gate's pinned sequence doesn't match)
DEV float qdiv(int a, int L) { return __double2float_rn(__ddiv_rn((double)a, (double)L)); }
__global__ void lred(const int* __restrict__ L, const int* __restrict__ E0, float* __restrict__ sp, float* __restrict__ sz, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int l = __ldg(L + i);
    sp[i] = qdiv(1, l);
    sz[i] = qdiv(__ldg(E0 + i), l);
  }
}

__global__ void hsref(const int* __restrict__ S, int8_t* __restrict__ P, float* __restrict__ sp, float* __restrict__ sz, int s, int rep,
                      int kvh, uint32_t A) {
  // any-s: launch (s / 128, rep) so pos is division-free (blockIdx.y the rep block, blockIdx.x the 128-wide query tile)
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= s) return;
  const int col = blockIdx.y * s + pos;
  const size_t N = (size_t)rep * s, qrow = (size_t)kvh * rep * s + col;
  int m = INT_MIN;
  for (int key = 0; key <= pos; key++) m = max(m, S[(size_t)key * N + col]);
  int L = 0, e0 = 0;
  int8_t* pr = P + qrow * s;
  for (int key = 0; key < s; key++) {
    int e = 0;
    if (key <= pos) e = fa_lut[min(__umulhi((uint32_t)(m - S[(size_t)key * N + col]), A), (uint32_t)(FA_LUT - 1))];
    L += e;
    if (key == 0) e0 = e;
    pr[key] = key == 0 ? 0 : (int8_t)e;
  }
  sp[qrow] = qdiv(1, L);
  sz[qrow] = qdiv(e0, L);
}

// the words of a table's tiles, R against R2 (a block a tile)
__global__ void rtcmp(const uint32_t* R, const uint32_t* R2, const int4* tab, int ntiles, int tiles_n, size_t tpk,
                      unsigned long long* bad) {
  unsigned long long c = 0;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
    const int4 e = tab[t];
    const size_t off = ((size_t)e.z * tpk + (size_t)e.x * tiles_n + e.y) * MLW::NCT * 8;
    for (int i = threadIdx.x; i < MLW::NCT * 8; i += blockDim.x) c += R[off + i] != R2[off + i];
  }
  if (c) atomicAdd(bad, c);
}
// P_prob's bytes apv reads: row (h, pos), keys below 128 (pos / 128 + 1)
__global__ void pcmp(const int8_t* P, const int8_t* P2, int s, int H, unsigned long long* bad) {
  unsigned long long c = 0;
  const size_t hb = (size_t)blockIdx.y * s;   // any-s: blockIdx.y the query head, blockIdx.x the position (so pos is division-free)
  for (int pos = blockIdx.x; pos < s; pos += gridDim.x) {
    const size_t r = hb + pos;
    const int nk = (pos | 127) + 1;
    for (int k = threadIdx.x; k < nk; k += blockDim.x) c += P[r * s + k] != P2[r * s + k];
  }
  if (c) atomicAdd(bad, c);
}

// ---------------------------------------------------------------- host
struct FA {
  AP V;
  int s = 0, H = 0, rep = 1, kvh = 0, ntc = 0, ntf = 0, grid = 0;
  size_t tpk = 0;
  uint32_t seed = 0, inst = 0, A = 0;
  float inv7 = 0.5f;
  bool h1 = false;    // cohalf H1's QK^T (co32c-u16): Q/K floored at top 111, the digit-split tail
  uint16_t* x;
  float2* qa = nullptr;   // cohalf H1's per (head, token) (max(x), max(-x)), filled by hrow_s for hquant_h1
  int8_t *W7, *X, *PA, *Wp, *PB, *Wh;
  int *M, *L, *E0, *S = nullptr;
  uint32_t *R, *R2 = nullptr;
  int4 *tabc, *tabf;
  std::vector<int4> htc;
  Maps *mq, *ms;
  std::vector<Maps*> mk;
};
// A = rint(sigma log2(e) 16 2^32), sigma = 2e-4 a unit of S (the logits' scale; a 7-bit X W^ has std about 15,500)
static uint32_t fa_A() { return (uint32_t)llrint(2e-4 * 1.4426950408889634 * 16.0 * 4294967296.0); }
static void fa_lut_host(uint8_t* lut) {
  for (int t = 0; t < FA_LUT; t++) lut[t] = (uint8_t)lrint(127.0 * exp2(-t / 16.0));
}
static void fasetup(FA& F, int s, int H, int rep, uint32_t seed, bool gate, bool unf) {
  if (s % 128 && unf) { fprintf(stderr, "s %d: the unfused h_l7qs needs s %% 128 == 0 (EL's raster; any-s: the last group is ragged)\n", s); exit(2); }
  apsetup(F.V, s, H, rep, seed, gate);
  F.s = s; F.H = H; F.rep = rep; F.kvh = H / rep; F.seed = seed; F.A = fa_A();
  const char* num = getenv("CG_NUMERICS");
  F.h1 = num && !strcmp(num, "h1");
  const int tq = s / 128, tn = rep * s / 128;
  F.tpk = (size_t)tq * tn;
  std::vector<int4> hf;
  for (int k = 0; k < F.kvh; k++)
    for (int n = 0; n < tn; n++)
      for (int m = 0; m < tq; m++) {
        hf.push_back(make_int4(m, n, k, n % tq));   // .w = the query tile's position within its rep block (any-s: so the device needs no modulo)
        if (m <= n % tq) F.htc.push_back(make_int4(m, n, k, n % tq));
      }
  F.ntc = (int)F.htc.size(); F.ntf = (int)hf.size();
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  F.grid = nsm;
  const size_t kr = (size_t)F.kvh * s, qr = (size_t)H * s, nr = (size_t)F.kvh * F.tpk * MLW::NCT * 8;
  CK(cudaMalloc(&F.x, kr * 128 * 2)); CK(cudaMalloc(&F.X, kr * 128)); CK(cudaMalloc(&F.PA, kr * 128));
  if (F.h1) CK(cudaMalloc(&F.qa, kr * sizeof(float2)));
  CK(cudaMalloc(&F.W7, qr * 128)); CK(cudaMalloc(&F.Wp, qr * 128)); CK(cudaMalloc(&F.PB, qr * 128)); CK(cudaMalloc(&F.Wh, qr * 128));
  CK(cudaMalloc(&F.M, qr * 4)); CK(cudaMalloc(&F.L, qr * 4)); CK(cudaMalloc(&F.E0, qr * 4));
  CK(cudaMalloc(&F.R, nr * 4));
  if (unf) {
    CK(cudaMalloc(&F.R2, nr * 4));
    CK(cudaMalloc(&F.S, (size_t)F.kvh * s * rep * s * 4));
  }
  CK(cudaMalloc(&F.tabc, F.ntc * sizeof(int4))); CK(cudaMalloc(&F.tabf, F.ntf * sizeof(int4)));
  CK(cudaMemcpy(F.tabc, F.htc.data(), F.ntc * sizeof(int4), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(F.tabf, hf.data(), F.ntf * sizeof(int4), cudaMemcpyHostToDevice));
  uint8_t lut[FA_LUT];
  fa_lut_host(lut);
  CK(cudaMemcpyToSymbol(fa_lut, lut, FA_LUT));
  xfill<<<4096, 256, 0, F.V.st>>>(F.x, kr * 128, mix32(seed ^ 0x3333u), 128, gate);
  // the query: cohalf H1's pre-bit Q in [-112, 110] (floor top 111); the 7-bit arm's in [-64, 62]
  const int wlo = F.h1 ? -112 : -64, whi = F.h1 ? 110 : 62;
  wfill<<<4096, 256, 0, F.V.st>>>(F.W7, qr * 128, mix32(seed ^ 0x6666u), 128, gate, wlo, whi);
  F.mq = upload_maps(Maps{tmap(F.X, 128, kr, 128), tmap(F.Wp, 128, qr, 128), tmap(F.PA, 128, kr, 128), tmap(F.PB, 128, qr, 128)});
  F.ms = upload_maps(Maps{tmap(F.Wh, 128, qr, 128), tmap(F.X, 128, kr, 128), tmap(F.Wh, 128, qr, 128), tmap(F.X, 128, kr, 128)});
  if (unf)
    for (int k = 0; k < F.kvh; k++) {
      const size_t ko = (size_t)k * s * 128, qo = (size_t)k * rep * s * 128;
      F.mk.push_back(upload_maps(Maps{tmap(F.X + ko, 128, s, 128), tmap(F.Wp + qo, 128, (size_t)rep * s, 128), tmap(F.PA + ko, 128, s, 128),
                                      tmap(F.PB + qo, 128, (size_t)rep * s, 128)}));
    }
  CK(cudaFuncSetAttribute(hm_l7qs, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  CK(cudaFuncSetAttribute(hm_l7qs_h1, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  CK(cudaFuncSetAttribute(hsm, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  CK(cudaFuncSetAttribute(h_l7qs, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  CK(cudaFuncSetAttribute(h_l7qs_h1, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  CK(cudaStreamSynchronize(F.V.st));
  CK(cudaGetLastError());
}
static void fafree(FA& F) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)F.x, (void*)F.qa, (void*)F.X, (void*)F.PA, (void*)F.W7, (void*)F.Wp, (void*)F.PB, (void*)F.Wh, (void*)F.M, (void*)F.L,
                  (void*)F.E0, (void*)F.S, (void*)F.R, (void*)F.R2, (void*)F.tabc, (void*)F.tabf, (void*)F.mq, (void*)F.ms})
    if (p) CK(cudaFree(p));
  for (Maps* m : F.mk) CK(cudaFree(m));
  apfree(F.V);
}
static void fq(FA& F, const Sig& sg) {
  const int kr = F.kvh * F.s, qr = F.H * F.s;
  const uint8_t* W = reinterpret_cast<const uint8_t*>(F.W7);
  uint8_t *Wp = reinterpret_cast<uint8_t*>(F.Wp), *PB = reinterpret_cast<uint8_t*>(F.PB), *Wh = reinterpret_cast<uint8_t*>(F.Wh);
  if (F.h1) {
    hrow_s<<<(kr + 7) / 8, 256, 0, F.V.st>>>(F.x, F.qa, kr);
    hquant_h1<<<hq_blocks(kr), 256, 0, F.V.st>>>(F.x, F.X, F.PA, kr, F.qa, sg);
    hwedit<true, true><<<hw_blocks(qr), 256, 0, F.V.st>>>(W, Wp, PB, qr, sg, Wh);
  } else {
    hquant<<<hq_blocks(kr), 256, 0, F.V.st>>>(F.x, F.X, F.PA, kr, F.inv7, sg);
    hwedit<true><<<hw_blocks(qr), 256, 0, F.V.st>>>(W, Wp, PB, qr, sg, Wh);
  }
}
static FaArgs faargs(FA& F, const Maps* m, bool full, uint32_t* R) {
  return FaArgs{m, full ? F.tabf : F.tabc, full ? F.ntf : F.ntc, F.s, F.rep, F.seed, F.A, R, F.M, F.L, F.E0, F.V.P};
}
static void fhm(FA& F, bool full, uint32_t* R) {
  CK(cudaMemsetAsync(F.M, 0x80, (size_t)F.H * F.s * 4, F.V.st));
  const int g = std::min(F.grid, full ? F.ntf : F.ntc);
  if (F.h1) hm_l7qs_h1<<<g, MLW::NT, CSMEM, F.V.st>>>(faargs(F, F.mq, full, R));
  else hm_l7qs<<<g, MLW::NT, CSMEM, F.V.st>>>(faargs(F, F.mq, full, R));
}
static void fsm(FA& F) {
  CK(cudaMemsetAsync(F.L, 0, (size_t)F.H * F.s * 4, F.V.st));
  hsm<<<std::min(F.grid, F.ntc), MLW::NT, CSMEM, F.V.st>>>(faargs(F, F.ms, false, nullptr));
}
static void flr(FA& F) { lred<<<(F.H * F.s + 255) / 256, 256, 0, F.V.st>>>(F.L, F.E0, F.V.sp, F.V.sz, F.H * F.s); }
static void fqks(FA& F) { fhm(F, false, F.R); fsm(F); flr(F); }
static void fpv(FA& F, const Sig& sg) { ued(F.V, sg); ved(F.V, sg); vsm(F.V); aprun(F.V, apv_fu, F.V.R3); }
// pv with apv.cu's vfold (X, DG, Sq and X(0) in one pass over V) for vedit and vsum
static void fpv_vf(FA& F, const Sig& sg) { ued(F.V, sg); vfd(F.V, sg); aprun(F.V, apv_fu, F.V.R3); }
// the unfused QK^T: h_l7qs a KV head at a time, O (= S^T) to S
static void fhqk(FA& F, uint32_t* R) {
  const size_t nk = (size_t)F.tpk * MLW::NCT * 8, sk = (size_t)F.s * F.rep * F.s;
  for (int k = 0; k < F.kvh; k++) {
    Args a{F.mk[k], F.S + k * sk, R + k * nk, F.s, F.rep * F.s, 128, 8, fa_seed(F.seed, k), 0, 1, 0, 0, {0, 0, 0, 0}, nullptr, 0};
    if (F.h1) h_l7qs_h1<<<std::min(F.grid, (int)F.tpk), MLW::NT, CSMEM, F.V.st>>>(a);
    else h_l7qs<<<std::min(F.grid, (int)F.tpk), MLW::NT, CSMEM, F.V.st>>>(a);
  }
}
static void fsref(FA& F) {
  const size_t sk = (size_t)F.s * F.rep * F.s;
  for (int k = 0; k < F.kvh; k++)
    hsref<<<dim3(F.s / 128, F.rep), 128, 0, F.V.st>>>(F.S + k * sk, F.V.P, F.V.sp, F.V.sz, F.s, F.rep, k, F.A);
}
// an arm: hquant (with hwedit<true>: the QK^T operands), hm (causal), hm_full (every tile), hsm (with L's memset),
// lred, qks (hm, hsm, lred: QK^T and the softmax, S never stored), pv (uedit, vedit, vsum, apv_fu), attn (the operands,
// qks and pv under one key), pv_vf and attn_vf (pv's and attn's with vfold for vedit and vsum), fu (apv_fu alone), hqk
// (the unfused h_l7qs, S stored)
static std::function<void(FA&)> farm(const std::string& a) {
  if (a == "hquant") return [](FA& F) { fq(F, sigma(F.seed, F.inst++)); };
  if (a == "hm") return [](FA& F) { fhm(F, false, F.R); };
  if (a == "hm_full") return [](FA& F) { fhm(F, true, F.R); };
  if (a == "hsm") return [](FA& F) { fsm(F); };
  if (a == "lred") return [](FA& F) { flr(F); };
  if (a == "qks") return [](FA& F) { fqks(F); };
  if (a == "pv") return [](FA& F) { fpv(F, sigma(F.seed, F.inst++)); };
  if (a == "pv_vf") return [](FA& F) { fpv_vf(F, sigma(F.seed, F.inst++)); };
  if (a == "fu") return [](FA& F) { aprun(F.V, apv_fu, F.V.R3); };
  if (a == "attn") return [](FA& F) {
    const Sig sg = sigma(F.seed, F.inst++);
    fq(F, sg); fqks(F); fpv(F, sg);
  };
  if (a == "attn_vf") return [](FA& F) {
    const Sig sg = sigma(F.seed, F.inst++);
    fq(F, sg); fqks(F); fpv_vf(F, sg);
  };
  if (a == "hqk") return [](FA& F) {
    if (!F.S) { fprintf(stderr, "hqk needs FA_UNF=1\n"); exit(2); }
    fhqk(F, F.R2);
  };
  fprintf(stderr, "no arm %s\n", a.c_str());
  exit(2);
}

static int fagate(const std::string& dir, int s, int H, int rep, uint32_t seed) {
  FA F;
  fasetup(F, s, H, rep, seed, true, true);
  AP& V = F.V;
  const size_t qr = (size_t)H * s, nr = (size_t)F.kvh * F.tpk * MLW::NCT * 8, ps = qr * s, fw = qr * 128;
  const size_t rw = (size_t)V.ntiles * MLW::NCT * 8;
  for (auto [p, b] : {std::pair<void*, size_t>{F.X, (size_t)F.kvh * s * 128}, {F.Wp, qr * 128}, {F.Wh, qr * 128}, {F.R, nr * 4},
                      {F.R2, nr * 4}, {F.S, (size_t)F.kvh * s * rep * s * 4}, {V.P, ps}, {V.sp, qr * 4}, {V.sz, qr * 4}, {V.F, fw * 4},
                      {V.R, rw * 4}, {V.R3, rw * 4}, {F.E0, qr * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, V.st));
  const Sig sg = sigma(seed, 0);
  fq(F, sg);
  // unfused: h_l7qs (S stored), hsref, the four producers (pedit on hsref's P_prob) and apv_l7qs
  fhqk(F, F.R2);
  fsref(F);
  producers(V, sg);
  aprun(V, apv_l7qs, V.R);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
  int8_t* Pr;
  float *spr, *szr;
  uint32_t* Rr;
  CK(cudaMalloc(&Pr, ps)); CK(cudaMalloc(&spr, qr * 4)); CK(cudaMalloc(&szr, qr * 4)); CK(cudaMalloc(&Rr, rw * 4));
  CK(cudaMemcpyAsync(Pr, V.P, ps, cudaMemcpyDeviceToDevice, V.st));
  CK(cudaMemcpyAsync(spr, V.sp, qr * 4, cudaMemcpyDeviceToDevice, V.st));
  CK(cudaMemcpyAsync(szr, V.sz, qr * 4, cudaMemcpyDeviceToDevice, V.st));
  CK(cudaMemcpyAsync(V.Fr, V.F, fw * 4, cudaMemcpyDeviceToDevice, V.st));
  CK(cudaMemcpyAsync(Rr, V.R, rw * 4, cudaMemcpyDeviceToDevice, V.st));
  // the control: poisoned, nothing run; every compare must fail
  for (auto [p, b] : {std::pair<void*, size_t>{F.R, nr * 4}, {V.P, ps}, {V.sp, qr * 4}, {V.sz, qr * 4}, {V.F, fw * 4}, {V.R3, rw * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, V.st));
  const int tiles_n = rep * s / 128;
  auto rbad = [&](const uint32_t* a, const int4* tab, int nt) {
    CK(cudaMemsetAsync(V.bad, 0, 8, V.st));
    rtcmp<<<1024, 256, 0, V.st>>>(a, F.R2, tab, nt, tiles_n, F.tpk, V.bad);
    unsigned long long h;
    CK(cudaMemcpyAsync(&h, V.bad, 8, cudaMemcpyDeviceToHost, V.st));
    CK(cudaStreamSynchronize(V.st));
    return h;
  };
  auto pbad = [&]() {
    CK(cudaMemsetAsync(V.bad, 0, 8, V.st));
    pcmp<<<dim3(std::min(1024, s), H), 256, 0, V.st>>>(V.P, Pr, s, H, V.bad);
    unsigned long long h;
    CK(cudaMemcpyAsync(&h, V.bad, 8, cudaMemcpyDeviceToHost, V.st));
    CK(cudaStreamSynchronize(V.st));
    return h;
  };
  const unsigned long long c_r = rbad(F.R, F.tabc, F.ntc), c_p = pbad(), c_sp = apbad(V, V.sp, spr, qr), c_sz = apbad(V, V.sz, szr, qr),
                           c_f = apbad(V, V.F, V.Fr, fw), c_pr = apbad(V, V.R3, Rr, rw);
  const bool control_rejected = c_r && c_p && c_sp && c_sz && c_f && c_pr;
  // fused: hm over every tile (receipts against h_l7qs's everywhere), then the causal path
  fhm(F, true, F.R);
  CK(cudaStreamSynchronize(V.st));
  const unsigned long long full_r = apbad(V, F.R, F.R2, nr);
  CK(cudaMemsetAsync(F.R, 0xA5, nr * 4, V.st));
  fqks(F);
  fpv(F, sg);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
  const unsigned long long r_bad = rbad(F.R, F.tabc, F.ntc), p_bad = pbad(), sp_bad = apbad(V, V.sp, spr, qr),
                           sz_bad = apbad(V, V.sz, szr, qr), f_bad = apbad(V, V.F, V.Fr, fw), pr_bad = apbad(V, V.R3, Rr, rw);
  // pv_vf from vfold's outputs and its own poisoned: f and receipts against the reference's
  const size_t vb = (size_t)V.kvh * 128 * s, dgb = (size_t)V.kvh * V.nJ * 128 * 128, sqb = (size_t)V.kvh * V.nJ * 128 * 4,
               x0b = (size_t)V.kvh * 128 * 4;
  for (auto [p, b] : {std::pair<void*, size_t>{V.Xt, vb}, {V.DG, dgb}, {V.Sq, sqb}, {V.X0, x0b}, {V.F, fw * 4}, {V.R3, rw * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, V.st));
  fpv_vf(F, sg);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
  const unsigned long long vf_f_bad = apbad(V, V.F, V.Fr, fw), vf_r_bad = apbad(V, V.R3, Rr, rw);
  const bool ok = full_r == 0 && r_bad == 0 && p_bad == 0 && sp_bad == 0 && sz_bad == 0 && f_bad == 0 && pr_bad == 0 && vf_f_bad == 0 &&
                  vf_r_bad == 0;
  // sampled query columns for fattn_ref.py: KV heads 0 and kvh - 1, X whole, and each sampled query's W^ row, S column
  // (h_l7qs's), P_prob row, s_q and z_q (the fused path's, equal to the reference's when ok), e(0) and m
  std::vector<int> ks = {0};
  if (F.kvh > 1) ks.push_back(F.kvh - 1);
  const int cols[] = {0, 1, 2, 31, 127, 128, 129, s / 2 - 1, s - 2, s - 1, s, s + 1, rep * s - 1, rep * s / 2 + 77};
  std::string cj;
  for (int k : ks) {
    dump(dir, ("X_" + std::to_string(k) + ".bin").c_str(), F.X + (size_t)k * s * 128, (size_t)s * 128, true);
    for (int col : cols) {
      if (col < 0 || col >= rep * s) continue;
      const size_t qrow = (size_t)k * rep * s + col;
      const std::string sfx = "_" + std::to_string(k) + "_" + std::to_string(col) + ".bin";
      dump(dir, ("Wh" + sfx).c_str(), F.Wh + qrow * 128, 128, true);
      dump(dir, ("P" + sfx).c_str(), V.P + qrow * s, s, true);
      std::vector<int> sc(s);
      CK(cudaMemcpy2D(sc.data(), 4, F.S + (size_t)k * s * rep * s + col, (size_t)rep * s * 4, 4, s, cudaMemcpyDeviceToHost));
      dump(dir, ("S" + sfx).c_str(), sc.data(), (size_t)s * 4, false);
      float spv, szv;
      int mv, e0v;
      CK(cudaMemcpy(&spv, V.sp + qrow, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&szv, V.sz + qrow, 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&mv, F.M + qrow, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&e0v, F.E0 + qrow, 4, cudaMemcpyDeviceToHost));
      char buf[200];
      snprintf(buf, sizeof buf, "%s[%d, %d, %.9g, %.9g, %d, %d]", cj.empty() ? "" : ", ", k, col, spv, szv, mv, e0v);
      cj += buf;
    }
  }
  uint8_t lut[FA_LUT];
  fa_lut_host(lut);
  std::string lj;
  for (int t = 0; t < FA_LUT; t++) lj += (t ? ", " : "") + std::to_string(lut[t]);
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"s\": %d, \"H\": %d, \"rep\": %d, \"seed\": %u, \"A\": %u, \"ntiles_causal\": %d, \"ntiles_full\": %d, "
              "\"cols\": [%s], \"lut\": [%s], \"device\": {\"hm_full_r_bad\": %llu, \"hm_r_bad\": %llu, \"p_bad\": %llu, "
              "\"sp_bad\": %llu, \"sz_bad\": %llu, \"f_bad\": %llu, \"pv_r_bad\": %llu, \"pv_vf_f_bad\": %llu, \"pv_vf_r_bad\": %llu}, "
              "\"control\": [%llu, %llu, %llu, %llu, %llu, %llu], \"device_ok\": %s, \"control_rejected\": %s}\n",
          s, H, rep, seed, F.A, F.ntc, F.ntf, cj.c_str(), lj.c_str(), full_r, r_bad, p_bad, sp_bad, sz_bad, f_bad, pr_bad, vf_f_bad, vf_r_bad,
          c_r, c_p, c_sp, c_sz, c_f, c_pr, ok ? "true" : "false", control_rejected ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"s\": %d, \"H\": %d, \"rep\": %d, \"device_ok\": %s, \"control_rejected\": %s, \"hm_full_r_bad\": %llu, "
         "\"hm_r_bad\": %llu, \"p_bad\": %llu, \"sp_bad\": %llu, \"sz_bad\": %llu, \"f_bad\": %llu, \"pv_r_bad\": %llu, "
         "\"pv_vf_f_bad\": %llu, \"pv_vf_r_bad\": %llu}\n",
         dir.c_str(), s, H, rep, ok ? "true" : "false", control_rejected ? "true" : "false", full_r, r_bad, p_bad, sp_bad, sz_bad, f_bad,
         pr_bad, vf_f_bad, vf_r_bad);
  for (void* p : {(void*)Pr, (void*)spr, (void*)szr, (void*)Rr}) CK(cudaFree(p));
  fafree(F);
  return ok && control_rejected ? 0 : 1;
}

static int fabench(int reps, int s, int H, int rep) {
  Nv nv;
  const std::vector<std::string> names = split(getenv("FA_ARMS"), "attn,qks,pv,hquant,hm,hm_full,hsm,lred,fu");
  bool unf = false;
  for (auto& a : names) unf |= a == "hqk";
  FA F;
  fasetup(F, s, H, rep, 0x5EED0000u + s + H, false, unf);
  const Sig s0 = sigma(F.seed, 0xFFFFFFFFu);
  fq(F, s0); fqks(F); producers(F.V, s0);
  std::vector<std::function<void(FA&)>> arms;
  for (auto& a : names) arms.push_back(farm(a));
  const int NA = (int)arms.size();
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_arm = [&](int j, int calls) {
    CK(cudaEventRecord(e0, F.V.st));
    for (int c = 0; c < calls; c++) arms[j](F);
    CK(cudaEventRecord(e1, F.V.st));
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
  printf("{\"s\": %d, \"H\": %d, \"rep\": %d, \"kvh\": %d, \"ntiles_causal\": %d, \"ntiles_full\": %d, \"pv_ntiles\": %d, \"grid\": %d, "
         "\"arms\": {", s, H, rep, F.kvh, F.ntc, F.ntf, F.V.ntiles, F.grid);
  for (int j = 0; j < NA; j++) {
    printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", names[j].c_str(), median(ms[j]),
           median(mhz[j]), calls[j]);
    for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
    printf("]}");
  }
  printf("}}\n");
  fflush(stdout);
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  fafree(F);
  return 0;
}

static int faone(const std::string& arm, int s, int H, int rep) {
  FA F;
  fasetup(F, s, H, rep, 0x5EED0000u + s + H, false, arm == "hqk");
  const Sig s0 = sigma(F.seed, 0xFFFFFFFFu);
  fq(F, s0); fqks(F); producers(F.V, s0);
  auto f = farm(arm);
  for (int i = 0; i < 3; i++) f(F);
  CK(cudaStreamSynchronize(F.V.st));
  CK(cudaProfilerStart());
  f(F);
  CK(cudaStreamSynchronize(F.V.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"one\": \"%s\", \"s\": %d, \"H\": %d, \"rep\": %d, \"ntiles_causal\": %d, \"grid\": %d}\n", arm.c_str(), s, H, rep, F.ntc,
         F.grid);
  fafree(F);
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return fagate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 6 && !strcmp(argv[1], "bench")) return fabench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 6 && !strcmp(argv[1], "one")) return faone(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: fattn gate DIR S H REP SEED | bench REPS S H REP | one ARM S H REP\n");
  return 2;
}
