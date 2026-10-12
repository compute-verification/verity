// Sketch-then-hash on the RTX PRO 6000 (sm_120), fused into the word codec's warp-specialized FP8 GEMM: `sketch.py` is
// the reference.
//
// A consumer warp holds a task's store bytes in registers as the A operand of the FP8 MMA; the same registers (the
// encoded words before the decode, or the plain FP8 bytes) are the A operand of mma.m16n8k32.s8.s8.s32 against the
// window's ±1 signs (B), k/8 MMAs per 32-byte step, so each fragment row's int32 sketch accumulates in the warp's
// registers over a run of the tile's tasks (`sketch.py`: tile (G, kc) is row blocks G T .. G T + T - 1 at chunk kc).
// A run of RUN tasks (RUN divides T) is one unit of the warp-specialized schedule: RUN = T stores the tile's sketch, RUN < T
// adds it atomically (exact: int32 addition is order-free), so a small layer still spreads over the device.
//
// Signs: the B fragments' bits, packed per (tile task i, word group u, lane) as 4 u32 words (`sketch.sign_fragments`):
// register r = 2 (s k/8 + blk) + h of step s is word r / 6, its byte q's sign at bit 16 ((r mod 6) / 3) + 4 q + r mod 3,
// so a register is one byte permute of two constants under that half of the word (`sgn`).
//
// The tag: ro-sketch-blake3/v0, BLAKE3-keyed(K, u64be j ‖ the k int32 little-endian), K from the name's prefix; leaves
// (16 tags, one tile) and the tree are benchmarks/bandwidth/tagger.cu's.
#include "../word_codec/wc.cu"

namespace sk {
using namespace wc;

static const char* const NAME = "ro-sketch-blake3/v0";

#ifndef SK_CHAINS
#define SK_CHAINS 8
#endif
// SK_L2_EF = 1 loads the store with an L2 evict-first policy, so the stream does not evict what a later call reads back
// from the L2 (the sketches a deferred commit hashes, in a deployment's two rotating slots; the activations).
#ifndef SK_L2_EF
#define SK_L2_EF 1
#endif
// SK_FUSE_DEC = 1 puts the encoded store's sketch and decode in one basic block, so ptxas may interleave the sketch's MMA
// chains with the decode's integer latency (otherwise the runtime `decode` branch separates them).
#ifndef SK_FUSE_DEC
#define SK_FUSE_DEC 0
#endif
// SK_SW = 1 sketches the encoded store in a warp of its own (one more warp a CTA), which reads every consumer's stage
// before the producer refills it; the decode loop is latency-bound with idle issue slots, the consumers then only decode.
#ifndef SK_SW
#define SK_SW 0
#endif

__device__ __forceinline__ void bulk_load_ef(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}

template <int KR>
__host__ __device__ constexpr int steps() { return KR > 0 ? (8 * LANE_WORDS + 15) / 16 : TASK_TILES; }
// Independent accumulators a sign block, so a warp has about SK_CHAINS sketch MMA chains in flight rather than k/8.
template <int KS>
__host__ __device__ constexpr int chains() { return SK_CHAINS / (KS / 8) > 1 ? SK_CHAINS / (KS / 8) : 1; }
template <int KR, int KS>
__host__ __device__ constexpr int sgroups() { return ((2 * steps<KR>() * (KS / 8) + 5) / 6 + 3) / 4; }

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Sign register r: each selector nibble's bit e = r mod 3 picks 0xFF (bit 3 is 0, so no sign replication).
__device__ __forceinline__ uint32_t sgn(const uint32_t* sw, int r) {
  const uint32_t w = sw[r / 6], sel = (r % 6) < 3 ? w : w >> 16, e = r % 3;
  const uint32_t lo = e == 0 ? 0xFF01FF01u : e == 1 ? 0xFFFF0101u : 0x01010101u;
  const uint32_t hi = e == 0 ? 0xFF01FF01u : e == 1 ? 0xFFFF0101u : 0xFFFFFFFFu;
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(lo), "r"(hi), "r"(sel));
  return d;
}

// Step s's k/8 sign MMAs; sw holds the task's packed sign words (4 a group).
template <int KS>
__device__ __forceinline__ void sketch_step(int (&acc)[KS / 8][4], const uint32_t (&a)[4], const uint32_t* sw, int s) {
  constexpr int KB = KS / 8;
#pragma unroll
  for (int blk = 0; blk < KB; ++blk) {
    const int r = 2 * (s * KB + blk);
    mma_s8(acc[blk], a, sgn(sw, r), sgn(sw, r + 1));
  }
}

// ro-sketch-blake3/v0 of one vector: BLAKE3-keyed(K, u64be j ‖ its KS int32 little-endian); clear zeroes the vector once
// read (a later call's atomic runs add into it).
template <int KS>
__device__ __forceinline__ void sk_tag(int* __restrict__ src, uint64_t j, const bw::Key3& key, int clear,
                                       uint32_t* __restrict__ out) {
  constexpr int MSG = 2 + KS, BLOCKS = (MSG + 15) / 16;
  uint32_t w[KS];
  const uint4* s4 = reinterpret_cast<const uint4*>(src);
#pragma unroll
  for (int i = 0; i < KS / 4; ++i) {
    const uint4 q = s4[i];
    w[4 * i] = q.x; w[4 * i + 1] = q.y; w[4 * i + 2] = q.z; w[4 * i + 3] = q.w;
  }
  if (clear) {
    uint4* d4 = reinterpret_cast<uint4*>(src);
#pragma unroll
    for (int i = 0; i < KS / 4; ++i) d4[i] = make_uint4(0, 0, 0, 0);
  }
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key.k[i];
#pragma unroll
  for (int b = 0; b < BLOCKS; ++b) {
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int u = 16 * b + i;
      m[i] = u == 0 ? bw::bswap32((uint32_t)(j >> 32)) : u == 1 ? bw::bswap32((uint32_t)j) : u < MSG ? w[u - 2] : 0u;
    }
    const uint32_t len = b + 1 < BLOCKS ? 64 : 4 * (MSG - 16 * (BLOCKS - 1));
    const uint32_t flags =
        bw::B3_KEYED | (b == 0 ? bw::B3_CHUNK_START : 0) | (b + 1 == BLOCKS ? bw::B3_CHUNK_END | bw::B3_ROOT : 0);
    bw::b3_compress(cv, m, 0, len, flags, cv);
  }
  uint4* o = reinterpret_cast<uint4*>(out);
  o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
  o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
}

// b3_leaf over tags an earlier kernel wrote, the next block's tag words loaded before the current block's compression.
__device__ __forceinline__ void sk_leaf(const uint32_t* __restrict__ tags, uint64_t B, uint64_t leaf, const bw::Mid& mid,
                                        uint64_t* __restrict__ leaves) {
  const uint64_t first = leaf * 16;
  const int n = (int)(B - first < 16 ? B - first : 16);
  const int data = 1 + 4 * n, nb = (8 * data + 17 + 127) / 128;
  const uint2* T = reinterpret_cast<const uint2*>(tags + first * 8);
  auto load = [&](int blk, uint2 (&r)[16]) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i;
      r[i] = q >= 1 && q < data ? __ldg(T + q - 1) : make_uint2(0, 0);
    }
  };
  uint64_t s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = mid.h[i];
  uint2 nxt[16];
  load(0, nxt);
#pragma unroll 1
  for (int blk = 0; blk < nb; ++blk) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i;
      uint64_t v = 0;
      if (q == 0)
        v = leaf;
      else if (q < data)
        v = ((uint64_t)bw::bswap32(nxt[i].x) << 32) | bw::bswap32(nxt[i].y);
      else if (q == data)
        v = 0x80ULL << 56;
      if (q == 16 * nb - 1) v = (uint64_t)(128 + 8 + 32 * n) * 8;
      w[i] = v;
    }
    if (blk + 1 < nb) load(blk + 1, nxt);
    bw::compress(s, w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) leaves[leaf * 8 + i] = s[i];
}

// The commit inside the GEMM: `committers` warps a CTA hash a previous call's nvec sketches (sk, ks int32 each) into
// tags (32 bytes each, block jbase + v), and either their leaves (16 tags, one tile, a leaf) or, when ltags is set, the
// leaves of an earlier call's lnvec tags, llanes leaves a leaf warp.
struct SkCommit {
  bw::Key3 key;
  bw::Mid leafmid;
  int* sk;
  uint64_t nvec;
  uint64_t jbase;
  uint32_t* tags;
  uint64_t* leaves;
  uint32_t* ltags;
  uint64_t* lleaves;
  uint64_t lnvec;
  int llanes;
  int committers;
  int clear;
};

template <int KR, int NB8, int KS, bool SK>
__global__ void __launch_bounds__((WS_MAX_WARPS + SK_SW) * 32) sk_ws_kernel(const void* __restrict__ wstore,
                                                                 const uint8_t* __restrict__ x, float* __restrict__ y,
                                                                 int M, int N, int Kd, uint64_t base, int groups,
                                                                 int consumers, int per_consumer, int partials, int T,
                                                                 int RUN, const uint4* __restrict__ signs,
                                                                 int* __restrict__ sk, int decode, SkCommit cm) {
  constexpr int SB = ws_stage_bytes<KR>(), KB = KS / 8, SG = sgroups<KR, KS>(), MS = steps<KR>(), NA = chains<KS>();
  constexpr bool SKW = SK && KR > 0 && SK_SW > 0, CSK = SK && !SKW;  // the sketch in its own warp, or the consumers'
  const int stages = consumers * per_consumer;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t (*xs)[XS_STRIDE] = reinterpret_cast<uint8_t (*)[XS_STRIDE]>(smem + stages * SB);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + stages * SB + NB8 * 8 * XS_STRIDE);
  uint64_t* empty = full + stages;
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks, g0 = blockIdx.x / nchunks, nbs = N / 16;
  const int rpt = T / RUN, nruns = (nbs + T - 1) / T * rpt;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  for (int e = threadIdx.x; e < NB8 * 8 * TASK_COLS / 16; e += blockDim.x) {
    int b = e / (TASK_COLS / 16), c = e % (TASK_COLS / 16);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < M) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)kc * TASK_COLS) + c);
    *reinterpret_cast<uint4*>(&xs[b][16 * c]) = v;
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, SKW ? 2 : 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  // consumer c's u-th task: run R = g0 + (c + consumers (u / RUN)) groups, row block (R / rpt) T + (R % rpt) RUN + u % RUN
  auto task_nb = [&](int c, int u) -> int {
    const int R = g0 + (c + consumers * (u / RUN)) * groups;
    if (R >= nruns) return -1;
    const int nb = (R / rpt) * T + (R % rpt) * RUN + u % RUN;
    return nb < nbs ? nb : -2;
  };

  if (warp == consumers) {
    if (lane == 0) {
      uint64_t pol = 0;
      if (SK_L2_EF) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      for (int u = 0;; ++u) {
        bool live = false;
        for (int c = 0; c < consumers; ++c) {
          const int nb = task_nb(c, u);
          live |= nb != -1;
          if (nb < 0) continue;
          const int s = c * per_consumer + u % per_consumer;
          if (u >= per_consumer) mbar_wait(empty + s, (u / per_consumer - 1) & 1);
          const uint64_t q = (uint64_t)nb * nchunks + kc;
          mbar_expect_tx(full + s, SB);
          if (SK_L2_EF)
            bulk_load_ef(smem + s * SB, static_cast<const uint8_t*>(wstore) + q * SB, SB, full + s, pol);
          else
            bulk_load(smem + s * SB, static_cast<const uint8_t*>(wstore) + q * SB, SB, full + s);
        }
        if (!live) break;
      }
    }
    return;
  }
  if constexpr (SKW) {
    if (warp == consumers + 1 + cm.committers) {
      // every consumer's task in the consumers' order, a task's sketch added on its own (a run's first task of a whole
      // tile stores, so the result is the consumers' whatever RUN)
      for (int u = 0;; ++u) {
        bool live = false;
        for (int c = 0; c < consumers; ++c) {
          const int nb = task_nb(c, u);
          live |= nb != -1;
          if (nb < 0) continue;
          const int s = c * per_consumer + u % per_consumer, i = nb % T;
          uint32_t sw[4 * SG];
          const uint4* sp = signs + (size_t)i * SG * LANES + lane;
#pragma unroll
          for (int v = 0; v < SG; ++v) {
            const uint4 t = __ldg(sp + v * LANES);
            sw[4 * v] = t.x; sw[4 * v + 1] = t.y; sw[4 * v + 2] = t.z; sw[4 * v + 3] = t.w;
          }
          mbar_wait(full + s, (u / per_consumer) & 1);
          const unsigned long long* st = reinterpret_cast<const unsigned long long*>(smem + s * SB) + lane;
          uint64_t xw[LANE_WORDS];
#pragma unroll
          for (int w = 0; w < LANE_WORDS; ++w) xw[w] = st[w * LANES];
          __syncwarp();
          if (lane == 0) mbar_arrive(empty + s);
          int acc[NA][KB][4];
#pragma unroll
          for (int a = 0; a < NA; ++a)
#pragma unroll
            for (int b = 0; b < KB; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0;
#pragma unroll
          for (int st8 = 0; st8 < MS; ++st8) {
            const uint64_t w0 = xw[2 * st8], w1 = 2 * st8 + 1 < LANE_WORDS ? xw[2 * st8 + 1] : 0;
            const uint32_t a[4] = {(uint32_t)w0, (uint32_t)(w0 >> 32), (uint32_t)w1, (uint32_t)(w1 >> 32)};
            sketch_step<KS>(acc[st8 % NA], a, sw, st8);
          }
#pragma unroll
          for (int a = 1; a < NA; ++a)
#pragma unroll
            for (int b = 0; b < KB; ++b)
#pragma unroll
              for (int e = 0; e < 4; ++e) acc[0][b][e] += acc[a][b][e];
          const int R = g0 + (c + consumers * (u / RUN)) * groups;
          const size_t tl = (size_t)(R / rpt) * nchunks + kc;
          const bool store = RUN == T && u % RUN == 0;
#pragma unroll
          for (int b = 0; b < KB; ++b) {
            int* r0 = sk + (tl * 16 + g) * KS + 8 * b + 2 * tq;
            int* r1 = r0 + 8 * KS;
            if (store) {
              *reinterpret_cast<int2*>(r0) = make_int2(acc[0][b][0], acc[0][b][1]);
              *reinterpret_cast<int2*>(r1) = make_int2(acc[0][b][2], acc[0][b][3]);
            } else {
              atomicAdd(r0, acc[0][b][0]); atomicAdd(r0 + 1, acc[0][b][1]);
              atomicAdd(r1, acc[0][b][2]); atomicAdd(r1 + 1, acc[0][b][3]);
            }
          }
        }
        if (!live) break;
      }
      return;
    }
  }
  if (warp > consumers && cm.ltags) {
    // the leaves llanes tiles a warp (one a lane: a leaf is five serial SHA-512 compressions, so a lane takes one), those
    // warps spread over the CTAs; the tags over the other warps, a contiguous share each
    const uint64_t nw = (uint64_t)gridDim.x * cm.committers, L = cm.llanes;
    const uint64_t wid = (uint64_t)blockIdx.x * cm.committers + (warp - consumers - 1);
    const uint64_t lt = cm.lnvec / 16, nlb = (lt + L - 1) / L, nlw = min(nlb, nw), stride = nlw ? nw / nlw : 1;
    const bool leafw = nlw && wid % stride == 0 && wid / stride < nlw;
    if (leafw)
      for (uint64_t b = wid / stride; b < nlb; b += nlw)
        if (lane < L && L * b + lane < lt) sk_leaf(cm.ltags, cm.lnvec, L * b + lane, cm.leafmid, cm.lleaves);
    const uint64_t ntw = nw > nlw ? nw - nlw : nw;
    if (leafw && ntw != nw) return;
    const uint64_t tw = ntw == nw ? wid : wid - min(nlw, (wid + stride - 1) / stride);
    const uint64_t vs = (cm.nvec + ntw - 1) / ntw, v_end = min(cm.nvec, (tw + 1) * vs);
    for (uint64_t v = tw * vs + lane; v < v_end; v += 32)
      sk_tag<KS>(cm.sk + v * KS, cm.jbase + v, cm.key, cm.clear, cm.tags + 8 * v);
    return;
  }
  if (warp > consumers) {
    // a contiguous share of the tiles a warp, 32 at a time: the warp's lanes tag their vectors, then one lane a leaf, so
    // a warp's latency is about one leaf whatever its share
    const uint64_t ntiles = cm.nvec / 16, nw = (uint64_t)gridDim.x * cm.committers;
    const uint64_t wid = (uint64_t)blockIdx.x * cm.committers + (warp - consumers - 1);
    const uint64_t share = (ntiles + nw - 1) / nw, t_end = min(ntiles, (wid + 1) * share);
    for (uint64_t t0 = wid * share; t0 < t_end; t0 += 32) {
      const uint64_t v_end = min(t_end, t0 + 32) * 16;
      for (uint64_t v = t0 * 16 + lane; v < v_end; v += 32)
        sk_tag<KS>(cm.sk + v * KS, cm.jbase + v, cm.key, cm.clear, cm.tags + 8 * v);
      __syncwarp();
      if (t0 + lane < t_end) b3_leaf(cm.tags, cm.nvec, t0 + lane, cm.leafmid, cm.leaves);
      __syncwarp();
    }
    return;
  }

  const uint64_t t32 = KR > 0 ? mulmod(kTau, LANES) : 0;
  int acc_s[NA][KB][4];
  for (int u = 0;; ++u) {
    const int nb = task_nb(warp, u);
    if (nb == -1) break;
    if (u % RUN == 0) {
#pragma unroll
      for (int c = 0; c < NA; ++c)
#pragma unroll
        for (int b = 0; b < KB; ++b) acc_s[c][b][0] = acc_s[c][b][1] = acc_s[c][b][2] = acc_s[c][b][3] = 0;
    }
    if (nb >= 0) {
      const int s = warp * per_consumer + u % per_consumer, i = nb % T;
      const uint64_t q = (uint64_t)nb * nchunks + kc;
      uint32_t sw[CSK ? 4 * SG : 1];
      if constexpr (CSK) {
        const uint4* sp = signs + (size_t)i * SG * LANES + lane;
#pragma unroll
        for (int v = 0; v < SG; ++v) {
          const uint4 t = __ldg(sp + v * LANES);
          sw[4 * v] = t.x; sw[4 * v + 1] = t.y; sw[4 * v + 2] = t.z; sw[4 * v + 3] = t.w;
        }
      }
      mbar_wait(full + s, (u / per_consumer) & 1);
      float acc[NB8][4];
#pragma unroll
      for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      auto tile = [&](int tt, const uint32_t (&a)[4]) {
#pragma unroll
        for (int j = 0; j < NB8; ++j) {
          uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 4 * tq]);
          uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 16 + 4 * tq]);
          mma_e4m3(acc[j], a, b0, b1);
        }
      };
      if constexpr (KR == 0) {
        const uint4* st = reinterpret_cast<const uint4*>(smem + s * SB) + lane;
        uint4 w[TASK_TILES];
#pragma unroll
        for (int tt = 0; tt < TASK_TILES; ++tt) w[tt] = st[tt * LANES];
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + s);
#pragma unroll
        for (int tt = 0; tt < TASK_TILES; ++tt) {
          const uint32_t a[4] = {w[tt].x, w[tt].y, w[tt].z, w[tt].w};
          tile(tt, a);
          if constexpr (CSK) sketch_step<KS>(acc_s[tt % NA], a, sw, tt);
        }
      } else {
        const unsigned long long* st = reinterpret_cast<const unsigned long long*>(smem + s * SB) + lane;
        uint64_t xw[LANE_WORDS];
#pragma unroll
        for (int w = 0; w < LANE_WORDS; ++w) xw[w] = st[w * LANES];
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + s);
        auto sketch_raw = [&]() {
          if constexpr (CSK) {
#pragma unroll
            for (int st8 = 0; st8 < MS; ++st8) {
              const uint64_t w0 = xw[2 * st8], w1 = 2 * st8 + 1 < LANE_WORDS ? xw[2 * st8 + 1] : 0;
              const uint32_t a[4] = {(uint32_t)w0, (uint32_t)(w0 >> 32), (uint32_t)w1, (uint32_t)(w1 >> 32)};
              sketch_step<KS>(acc_s[st8 % NA], a, sw, st8);
            }
          }
        };
        if (SK_FUSE_DEC && decode) {
          sketch_raw();
          decode_lane<KR>(xw, mulmod(kTau, (base + q * TASK_WORDS + lane) % P), t32);
        } else {
          sketch_raw();
          if (decode) decode_lane<KR>(xw, mulmod(kTau, (base + q * TASK_WORDS + lane) % P), t32);
        }
#pragma unroll
        for (int tt = 0; tt < TASK_TILES; ++tt) {
          const uint32_t a[4] = {take4(xw, 128 * tt), take4(xw, 128 * tt + 32), take4(xw, 128 * tt + 64),
                                 take4(xw, 128 * tt + 96)};
          tile(tt, a);
        }
      }
#pragma unroll
      for (int j = 0; j < NB8; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int n = nb * 16 + g + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
          if (b < M) {
            if (partials) y[((size_t)kc * M + b) * N + n] = acc[j][e];
            else atomicAdd(y + (size_t)b * N + n, acc[j][e]);
          }
        }
      }
    }
    if (CSK && (u % RUN == RUN - 1)) {
      const int R = g0 + (warp + consumers * (u / RUN)) * groups;
      const size_t tl = (size_t)(R / rpt) * nchunks + kc;
#pragma unroll
      for (int c = 1; c < NA; ++c)
#pragma unroll
        for (int b = 0; b < KB; ++b)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc_s[0][b][e] += acc_s[c][b][e];
#pragma unroll
      for (int b = 0; b < KB; ++b) {
        int* r0 = sk + (tl * 16 + g) * KS + 8 * b + 2 * tq;
        int* r1 = r0 + 8 * KS;
        if (RUN == T) {
          *reinterpret_cast<int2*>(r0) = make_int2(acc_s[0][b][0], acc_s[0][b][1]);
          *reinterpret_cast<int2*>(r1) = make_int2(acc_s[0][b][2], acc_s[0][b][3]);
        } else {
          atomicAdd(r0, acc_s[0][b][0]); atomicAdd(r0 + 1, acc_s[0][b][1]);
          atomicAdd(r1, acc_s[0][b][2]); atomicAdd(r1 + 1, acc_s[0][b][3]);
        }
      }
    }
  }
}

// ro-sketch-blake3/v0 over vectors [0, nvec) of sk, one thread a vector.
template <int KS>
__global__ void __launch_bounds__(128) sk_hash_kernel(int* __restrict__ sk, uint64_t nvec, bw::Key3 key,
                                                      uint64_t jbase, uint32_t* __restrict__ tags, int clear) {
  const uint64_t v = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (v < nvec) sk_tag<KS>(sk + v * KS, jbase + v, key, clear, tags + 8 * v);
}

// K = SHA512(prefix(TAG_DOMAIN, NAME, seed))[0:32] as BLAKE3's little-endian key words (tagger.cu's b3_key, this name).
static bw::Key3 sk_key(const uint8_t seed[32]) {
  std::vector<uint8_t> p = bw::prefix_block(bw::TAG_DOMAIN, NAME, seed);
  uint64_t d[8];
  bw::sha512_host(p.data(), p.size(), d);
  bw::Key3 k;
  for (int i = 0; i < 8; ++i) {
    const uint64_t w = d[i / 2];
    const uint32_t be = (uint32_t)(i & 1 ? w : w >> 32);
    k.k[i] = (be >> 24) | ((be >> 8) & 0xff00u) | ((be << 8) & 0xff0000u) | (be << 24);
  }
  return k;
}

}  // namespace sk

#ifndef SK_KMAX
#define SK_KMAX 3
#endif

extern "C" {

int sk_constants(int* out) {
  out[0] = wc::LANES; out[1] = wc::LANE_WORDS; out[2] = wc::PB; out[3] = sk::steps<0>(); out[4] = sk::steps<1>();
  out[5] = SK_KMAX; out[6] = wc::WS_MAX_WARPS;
  return 0;
}

// The tag's key and the leaves' midstate for the last seed.
static sk::SkCommit sk_keys(const uint8_t* seed) {
  static uint8_t last[32];
  static bool have = false;
  static sk::SkCommit t{};
  if (!have || memcmp(last, seed, 32)) {
    memcpy(last, seed, 32);
    have = true;
    t.key = sk::sk_key(seed);
    t.leafmid = bw::prefix_midstate(bw::LEAF_DOMAIN, sk::NAME, seed);
  }
  return t;
}

// The fused GEMM with the sketch: ks in {8, 16, 32} (0: the same schedule without it); kr the codec's rounds (0: the plain
// store); T row blocks a tile, RUN tasks a run (RUN divides T; RUN < T needs sk zeroed); signs as sketch.sign_fragments for
// (ks, T); sk gets tile t's 16 × ks int32 at 16 ks t. groups <= 0: as many CTAs as fit (times -groups), split over the
// chunks. committers > 0 warps a CTA hash csk's cnvec sketches (a previous call's, ks each) into the commit workspace
// cws (wc_commit_bytes(cnvec)): tags at block cjbase + v and their leaves, zeroing csk behind them when clear.
int sk_gemm_deferred(int kr, int ks, const void* w, const uint8_t* x, float* y, int M, int N, int Kd, uint64_t base,
                     int partials, int consumers, int per_consumer, int groups, int* launched, int T, int RUN,
                     const void* signs, int* skout, int decode, const uint8_t* seed, int committers, int* csk,
                     uint64_t cnvec, uint64_t cjbase, void* cws, int clear, uint64_t lnvec, void* lws, int leaf_lanes,
                     cudaStream_t s);

int sk_gemm(int kr, int ks, const void* w, const uint8_t* x, float* y, int M, int N, int Kd, uint64_t base, int partials,
            int consumers, int per_consumer, int groups, int* launched, int T, int RUN, const void* signs, int* skout,
            int decode, const uint8_t* seed, int committers, int* csk, uint64_t cnvec, uint64_t cjbase, void* cws,
            int clear, cudaStream_t s) {
  return sk_gemm_deferred(kr, ks, w, x, y, M, N, Kd, base, partials, consumers, per_consumer, groups, launched, T, RUN,
                          signs, skout, decode, seed, committers, csk, cnvec, cjbase, cws, clear, 0, nullptr, 0, s);
}

// sk_gemm, but with lws (a commit workspace whose lnvec tags an earlier call wrote) the committer warps compute lws's
// leaves, leaf_lanes a warp, instead of the leaves of the tags they write; lnvec = 0: they only tag (cws may then point
// into a larger workspace, a later call computing its leaves).
int sk_gemm_deferred(int kr, int ks, const void* w, const uint8_t* x, float* y, int M, int N, int Kd, uint64_t base,
                     int partials, int consumers, int per_consumer, int groups, int* launched, int T, int RUN,
                     const void* signs, int* skout, int decode, const uint8_t* seed, int committers, int* csk,
                     uint64_t cnvec, uint64_t cjbase, void* cws, int clear, uint64_t lnvec, void* lws, int leaf_lanes,
                     cudaStream_t s) {
  if (M < 1 || M > 32 || N % 16 || Kd % wc::TASK_COLS || kr < 0 || kr > SK_KMAX || consumers < 1 ||
      consumers > wc::WS_MAX_CONSUMERS || per_consumer < 1 || consumers * per_consumer > 32 || T < 1 || RUN < 1 ||
      T % RUN || (ks && (!signs || !skout)) || (ks != 0 && ks != 8 && ks != 16 && ks != 32) || committers < 0 ||
      consumers + 1 + committers > wc::WS_MAX_WARPS ||
      (committers && (!ks || !seed || !csk || !cws || !cnvec || cnvec % 16)) ||
      (lws && (!committers || lnvec % 16 || lws == cws || (lnvec && (leaf_lanes < 1 || leaf_lanes > 32)))))
    return 1;
  sk::SkCommit cm{};
  if (committers) {
    cm = sk_keys(seed);
    bw::Pass p = bw::make_pass(bw::TAG_B3, cnvec, cws);
    cm.sk = csk;
    cm.nvec = cnvec;
    cm.jbase = cjbase;
    cm.tags = p.tags;
    cm.leaves = p.nodes;
    cm.committers = committers;
    cm.clear = clear;
    if (lws && lnvec) {
      bw::Pass l = bw::make_pass(bw::TAG_B3, lnvec, lws);
      cm.ltags = l.tags;
      cm.lleaves = l.nodes;
      cm.lnvec = lnvec;
      cm.llanes = leaf_lanes;
    } else if (lws) {  // tags only: the leaves are another call's
      cm.ltags = static_cast<uint32_t*>(lws);
      cm.llanes = 1;
    }
  }
  const int stages = consumers * per_consumer;
  const int nb8 = M <= 8 ? 1 : M <= 16 ? 2 : 4, nchunks = Kd / wc::TASK_COLS, nbs = N / 16;
  const int nruns = (nbs + T - 1) / T * (T / RUN);
  const int threads = (consumers + 1 + committers + (SK_SW && kr > 0 && ks ? 1 : 0)) * 32;
  int sms = 0, per_sm = 0, dev = 0;
  if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)) return 2;
  const uint4* sg = static_cast<const uint4*>(signs);
#define G(KR, NB, KS, SKB) do { \
    auto kern = sk::sk_ws_kernel<KR, NB, KS, SKB>; \
    const int smem = wc::ws_smem_bytes<KR, NB>(stages); \
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) return 2; \
    if (groups <= 0) { \
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, threads, smem) || per_sm < 1) return 2; \
      const int mult = groups < 0 ? -groups : 1; \
      groups = mult * sms * per_sm / nchunks > 0 ? mult * sms * per_sm / nchunks : 1; \
    } \
    if (groups > nruns) groups = nruns; \
    kern<<<groups * nchunks, threads, smem, s>>>(w, x, y, M, N, Kd, base, groups, consumers, per_consumer, partials, T, \
                                                 RUN, sg, skout, decode, cm); } while (0)
#define GS(KR, NB) do { switch (ks) { case 0: G(KR, NB, 8, false); break; case 8: G(KR, NB, 8, true); break; \
                                      case 16: G(KR, NB, 16, true); break; default: G(KR, NB, 32, true); } } while (0)
#define GK(KR) do { if (nb8 == 1) GS(KR, 1); else if (nb8 == 2) GS(KR, 2); else GS(KR, 4); } while (0)
  switch (kr) {
    case 0: GK(0); break;
    case 1: GK(1); break;
#if SK_KMAX >= 2
    case 2: GK(2); break;
#endif
#if SK_KMAX >= 3
    case 3: GK(3); break;
#endif
    default: return 1;
  }
#undef GK
#undef GS
#undef G
  if (launched) *launched = groups * nchunks;
  return cudaGetLastError() ? 4 : 0;
}

// The tags of vectors [0, nvec) of sk into tags (32 bytes each, tagger.cu's b3 layout), block index jbase + v; clear
// zeroes sk behind the read.
int sk_hash(int ks, int* skin, uint64_t nvec, const uint8_t* seed, uint64_t jbase, uint32_t* tags, int clear,
            cudaStream_t s) {
  const bw::Key3 key = sk_keys(seed).key;
  const unsigned grid = (unsigned)((nvec + 127) / 128);
  switch (ks) {
    case 8: sk::sk_hash_kernel<8><<<grid, 128, 0, s>>>(skin, nvec, key, jbase, tags, clear); break;
    case 16: sk::sk_hash_kernel<16><<<grid, 128, 0, s>>>(skin, nvec, key, jbase, tags, clear); break;
    case 32: sk::sk_hash_kernel<32><<<grid, 128, 0, s>>>(skin, nvec, key, jbase, tags, clear); break;
    default: return 1;
  }
  return cudaGetLastError() ? 4 : 0;
}

// The leaves of the workspace's tags (wc_commit_bytes(nvec) bytes) under this tag's name.
int sk_leaves(void* workspace, uint64_t nvec, const uint8_t* seed, cudaStream_t s) {
  bw::Pass p = bw::make_pass(bw::TAG_B3, nvec, workspace);
  const uint64_t leaves = (nvec + 15) / 16;
  bw::b3_leaves<<<(unsigned)((leaves + 127) / 128), 128, 0, s>>>(
      p.tags, nvec, nvec, sk_keys(seed).leafmid, p.nodes);
  return cudaGetLastError() ? 4 : 0;
}

}  // extern "C"
