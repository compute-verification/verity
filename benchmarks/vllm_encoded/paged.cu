// vLLM's paged KV cache in `benchmarks/word_codec/kv.cu`'s codec: decode attention over the pages, the append of a
// step's new tokens, and the gather of whole pages for prefill (FlashAttention reads them from a scratch). Build with
// -DKV_FIELD, -DKV_MASK, -DKV_FOLD6 and -DKV_QHEADS at the encoded stack's codec (run.sh).
//
// A layer's cache is vLLM's own tensor, [blocks][KV heads][slots][K ‖ V] or any order of the inner three whose block
// is contiguous: head g of slot s of block b, its 256 bytes of K then 256 of V, is HW words at physical word offset
// b sb + g sh + s ss. The codec index is independent of that physical layout: after the weights at store_base, physical
// slot u and layer l are the store-order/v0 block u layers + l, whose words are K of every head then V of every head.
// Thus a token-layer is exactly one registered 4 KiB block and all three paths agree on its codec indices. A token-layer
// written after the codec seed is encoded under its bound key: each path takes the layer's offsets bj, [physical
// slot][NL] (null: the seed's key), slot u's b_j under PoUB (`codec.binding` of the token-layer's digest d_j and
// its store index j), plus tau (j TLW - store_base - (u layers + l) TLW) mod p when j is not the slot's own block.
//
// A token-layer that PoUB appends is written plaintext and encoded in place once its append coin c_a arrives
// (deferred residency, remote_bandwidth's PROTOCOL.md §4), so an encoded cache may hold plaintext slots: each path takes
// the layer's flags plain, [physical slot] bytes (null: none), nonzero for a slot whose token-layer is still plaintext,
// which it reads as it is and gives no receipt (no record holds it yet). pg_append records in plain how it wrote each
// slot; pg_encode encodes listed plaintext slots in place under their bound offsets b_j(c_a) and clears their flags.
// A token-layer kept past the lifetime cap L is re-keyed in its slot (§4: appended again at a fresh index j' under a fresh
// coin): pg_rekey decodes listed slots under their old offsets and encodes them under their new ones b_j'(c_a') in place,
// then writes the new offsets over the old.
//
// pg_attn is kv.cu's warp-specialized tensor-core form (attn_mma_ws_kernel) over pages: MDW decoder warps, a lane a
// word of every tile, load the tile's words of one KV head from the pages (the block table), decode them (encoded) or
// pass them on (plain) into a ring of MNB tiles, and the MW MMA warps fold each tile in kv.cu's arithmetic
// (mma_chunk), so plain and encoded agree bit for bit. A sequence's tokens are split into `splits` ranges of whole
// tiles by its own length (lens), so the grid is fixed and a CUDA graph replays it at any lengths; an empty split
// leaves no tokens (max -1e30, sum 0), a sequence of length 0 a zero row. q is read from vLLM's query (bf16, any row
// stride) and scaled by qscale (log2(e) / sqrt(128)) and rounded to bf16 in place of kv.cu's pre-scaled q; the output
// is bf16, vLLM's. With ck (and cv) a sequence's last token, position len − 1, is read from the step's own K and V rows
// (bf16, vLLM's key and value, the sequence's row as q's) and not from the pages: the stack appends a step's tokens after
// its forward (pg_append_multi, on a second stream), so its pages hold them only from the next step.
//
// pg_append_multi is pg_append over many layers in one launch (the codec's encode is a long chain a word: one wave for all
// of a step's layers instead of one a layer), each layer's rows, slots, cache and first index in a table on the device.
//
// pg_window is a decode step's windows for an attention that reads each one whole (the exact-int8 attention's
// Int8KvQuant and Int8LutAttentionRow, `verity_vllm.engine.int8_attention`'s window_of): request r's tokens 0 … lens[r] − 1,
// K and V as bf16, into a working buffer at rows wo[r] + t (pg_window_offsets writes wo, the prefix sums of lens), head g
// of row R at R ots + g ogs elements, each word decoded (encoded; a plaintext slot as it is) or copied, and the last token
// from ck, cv as pg_attn takes it. It gives no receipts. Its grid is fixed (every SM's resident CTAs, each striding over
// the rows) and every length, offset and table is read on the device, so a CUDA graph replays it at any lengths. order is
// the codec's word order in a token-layer: 0 pg_wi's, K of every head then V of every head; 1 each head's K ‖ V, the
// physical order of a [pages][slots][heads][K ‖ V] cache, which registered a layer to a region (layers 1, store_base the
// region's first word) has codec index store_base plus its physical word.
//
// pg_reduce is the stack's linear epilogue: wt.cu's chunk partial sums [nch][M][N] summed in chunk order and scaled
// (bf16(sum * xs[row or 0] * ws)), so a plain and an encoded linear over the same weight give the same bits.
#include "../word_codec/kv.cu"

#include <cuda_bf16.h>

namespace kv {

struct Pg {
  const int *lens;   // [batch] tokens a sequence
  const int *bt;     // [batch][bstride] block ids
  const int *qrow;   // [batch] its row of q and of out (nullptr: the sequence's index)
  int bstride, bs;   // the block table's row stride, slots a block
  int sb, sh, ss;    // words between blocks, heads, slots
  u64 store_base;    // the KV region's first word index, after the weights
  int layers, layer;
  const uint16_t *ck, *cv;  // the step's K and V rows (nullptr: every token from the pages)
  size_t cks, cvs;          // their row strides (elements)
  int *rc_sk;               // sparse [physical slot][KV head][16] shares, nullptr: no receipts
  int *used;                // sparse physical-slot epochs
  const int *epoch;
  const u32 *bj;            // [physical slot][NL] this layer's token-layers' bound offsets, nullptr: the seed's key
  const uint8_t *plain;     // [physical slot] nonzero: the token-layer is plaintext, nullptr: every slot encoded
};

__device__ __forceinline__ bool pg_plain(const uint8_t *plain, u64 slot) { return plain && plain[slot]; }

__device__ __forceinline__ u64 pg_word(const Pg &pg, const int *btr, int p, int g, int j) {
  return (u64)btr[p / pg.bs] * pg.sb + (u64)g * pg.sh + (u64)(p % pg.bs) * pg.ss + j;
}

__device__ __forceinline__ u64 pg_slot(const Pg &pg, const int *btr, int p) {
  return (u64)btr[p / pg.bs] * pg.bs + p % pg.bs;
}

__device__ __forceinline__ int pg_wi(int g, int j) {
  return (j < HW / 2 ? 0 : TL_BYTES / 2 / WB - HW / 2) + g * (HB / WB) + j;
}

__device__ __forceinline__ u64 pg_store_word(const Pg &pg, u64 slot, int g, int j) {
  return pg.store_base + (slot * pg.layers + pg.layer) * TLW + pg_wi(g, j);
}

__device__ __forceinline__ u32 pg_scale2(u32 v, float s) {
  float a, b;
  return bf16x2_rn(__uint_as_float(v << 16) * s, __uint_as_float(v & 0xffff0000u) * s, a, b);
}

// mma_load_q's fragments from vLLM's query row (bf16, unscaled), each element times s rounded to bf16.
__device__ __forceinline__ void pg_load_q(const uint16_t *q, size_t qstride, size_t row, int g, int lane, float s,
                                          u32 (&qa)[8][2]) {
  const int gid = lane >> 2, t4 = lane & 3;
  const u32 *qp = reinterpret_cast<const u32 *>(q + row * qstride + (size_t)(g * GROUP + (gid < GROUP ? gid : 0)) * HEAD_DIM);
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    qa[ks][0] = gid < GROUP ? pg_scale2(qp[8 * ks + t4], s) : 0u;
    qa[ks][1] = gid < GROUP ? pg_scale2(qp[8 * ks + 4 + t4], s) : 0u;
  }
}

constexpr int pg_smem() { return MNB * MT * MTS + 2 * MNB * 8; }
constexpr int pg_rc_smem() { return pg_smem() + RK * 32 * 16; }

template <bool ENC, bool RC>
__global__ void __launch_bounds__(MWT, 2) pg_attn_kernel(const uint8_t *__restrict__ kv, const uint16_t *__restrict__ q,
                                                         size_t qstride, float qscale, float *__restrict__ part, Pg pg,
                                                         int splits, int k) {
  if constexpr (MMA_OK) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint64_t *full = reinterpret_cast<uint64_t *>(sm + MNB * MT * MTS), *empty = full + MNB;
    u32 *rct = reinterpret_cast<u32 *>(empty + MNB);
    const int g = blockIdx.x % KV_HEADS, sq = blockIdx.x / KV_HEADS, seq = sq / splits, split = sq % splits;
    const int len = pg.lens[seq];
    const int per = ((len + splits - 1) / splits + MT - 1) / MT * MT;
    const int t0 = min(len, split * per), t1 = min(len, t0 + per);
    const int ntiles = (t1 - t0 + MT - 1) / MT;
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
      for (int b = 0; b < MNB; ++b) {
        mbar_init(full + b, 32 * MDW);
        mbar_init(empty + b, 32 * MW);
      }
    }
    if constexpr (RC)
      for (int v = threadIdx.x; v < RK * 32; v += MWT)
        reinterpret_cast<uint4 *>(rct)[v] = kRcA[g][v / 32][v % 32];
    __syncthreads();
    if (w >= MW) {
      const int d = threadIdx.x - 32 * MW, t = d / HW, j = d % HW;
      const int *btr = pg.bt + (size_t)seq * pg.bstride;
      const u32 *base = reinterpret_cast<const u32 *>(kv);
      const size_t crow = pg.qrow ? (size_t)pg.qrow[seq] : (size_t)seq;
      const u32 *fresh = pg.ck ? reinterpret_cast<const u32 *>(
                                     (j < HW / 2 ? pg.ck + crow * pg.cks : pg.cv + crow * pg.cvs) + (size_t)g * HEAD_DIM +
                                     (size_t)(j % (HW / 2)) * (WB / 2))
                               : nullptr;
      const auto has = [&](int tile) { return tile < ntiles && t0 + tile * MT + t < t1; };
      const auto own = [&](int tile) { return fresh && t0 + tile * MT + t == len - 1; };
      u32 x[NL];
      u64 wl = has(0) ? pg_word(pg, btr, t0 + t, g, j) : 0;
      u64 sl = has(0) ? pg_slot(pg, btr, t0 + t) : 0;
      bool mine = has(0) && own(0);
      if (has(0)) {
        if (mine) load(fresh, x);
        else ldg_word(base + wl * NL, x);
      }
      for (int tile = 0; tile < ntiles; ++tile) {
        const bool cur = has(tile), next = has(tile + 1), next_mine = next && own(tile + 1);
        const u64 wn = next ? pg_word(pg, btr, t0 + (tile + 1) * MT + t, g, j) : 0;
        const u64 sn = next ? pg_slot(pg, btr, t0 + (tile + 1) * MT + t) : 0;
        if (next && !next_mine) prefetch_word_l2(base + wn * NL);
        if (!cur) {
#pragma unroll
          for (int l = 0; l < NL; ++l) x[l] = 0u;
        }
        if constexpr (RC) {
          int c[4] = {};
          const uint4 *a = reinterpret_cast<const uint4 *>(rct) + lane;
#pragma unroll
          for (int kk = 0; kk < RK; ++kk) {
            const uint4 v = a[32 * kk];
            imma16832(c, v.x, v.y, v.z, v.w, x[2 * kk], x[2 * kk + 1]);
          }
          const int i = w - MW, gid = lane >> 2, t4 = lane & 3;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int n = 2 * t4 + (e & 1), pt = t0 + tile * MT + 8 * i + n;
            if (pt < t1 && !(pg.ck && pt == len - 1))
              pg.rc_sk[((pg_slot(pg, btr, pt) * KV_HEADS + g) * 16) + gid + 8 * (e >> 1)] = c[e];
          }
          if (cur && !mine && g == 0 && j == 0 && !pg_plain(pg.plain, sl)) atomicExch(pg.used + sl, *pg.epoch);
        }
        if constexpr (ENC) {
          if (cur && !mine && !pg_plain(pg.plain, sl)) {
            u32 tw[NL];
            tweak_at(pg_store_word(pg, sl, g, j), pg.bj ? pg.bj + sl * NL : nullptr, tw);
            decode_word_t<true>(x, tw, k);
          }
        }
        if (tile >= MNB) mbar_wait(empty + tile % MNB, (tile / MNB - 1) & 1);
        store(reinterpret_cast<u32 *>(sm + (tile % MNB) * MT * MTS + t * MTS + j * WB), x);
        mbar_arrive(full + tile % MNB);
        if (next) {
          if (next_mine) load(fresh, x);
          else ldg_word(base + wn * NL, x);
        }
        wl = wn;
        sl = sn;
        mine = next_mine;
      }
      return;
    }
    u32 qa[8][2];
    float o[8][4] = {};
    float m_own = -INFINITY, l_own = 0.f;
    pg_load_q(q, qstride, pg.qrow ? (size_t)pg.qrow[seq] : (size_t)seq, g, lane, qscale, qa);
    const int rem = t1 - t0;
    for (int tile = 0; tile * MT < rem; ++tile) {
      const int b = tile % MNB, nt = min(MT, rem - tile * MT);
      mbar_wait(full + b, (tile / MNB) & 1);
      if (nt > 16 * w)
        mma_chunk(smem_u32(sm + b * MT * MTS + 16 * w * MTS), qa, lane, min(16, nt - 16 * w), o, m_own, l_own);
      mbar_arrive(empty + b);
    }
    mma_out(part + ((size_t)sq * MW + w) * PART, g, lane, o, m_own, l_own);
  }
}

// One selected physical slot's standard receipt for this layer, into the sparse token-layer tag array. The attention
// and gather kernels have already written each KV head's share of σ while holding the encoded words.
__global__ void __launch_bounds__(128) pg_receipt_sparse_kernel(const int *__restrict__ sk, u32 *__restrict__ tags,
                                                                const int *__restrict__ used, const int *__restrict__ epoch,
                                                                u64 slots, int layers, int layer, u64 jbase) {
  const u64 u = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= slots || used[u] != *epoch) return;
  int sg[16] = {};
  const int4 *p = reinterpret_cast<const int4 *>(sk + u * KV_HEADS * 16);
#pragma unroll
  for (int g = 0; g < KV_HEADS; ++g)
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int4 q = __ldcs(p + 4 * g + v);
      sg[4 * v] += q.x, sg[4 * v + 1] += q.y, sg[4 * v + 2] += q.z, sg[4 * v + 3] += q.w;
    }
  u64 h[8], w[16];
  w[0] = jbase + u * layers + layer;
#pragma unroll
  for (int i = 0; i < 8; ++i) w[1 + i] = be64((u32)sg[2 * i], (u32)sg[2 * i + 1]);
  w[9] = 0x8000000000000000ull;
#pragma unroll
  for (int i = 10; i < 15; ++i) w[i] = 0;
  w[15] = (128 + 72) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] = kKvMid[i];
  sha512_compress(h, w);
  uint4 *o = reinterpret_cast<uint4 *>(tags + 16 * (u * layers + layer));
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = u64_pair(h[2 * i], h[2 * i + 1]);
}

// attn_combine's merge of a sequence's nparts partials, written as bf16 at its row of out; a row with no tokens (sum 0)
// and every row past batch (a CUDA graph's padding) is zero.
__global__ void __launch_bounds__(32 * CW) pg_combine(const float *__restrict__ part, uint16_t *__restrict__ out,
                                                      size_t ostride, const int *__restrict__ orow, int batch,
                                                      int nparts) {
  __shared__ float4 sacc[CW][32];
  __shared__ float sm_[CW], sl[CW];
  const int seq = blockIdx.x / HEADS, h = blockIdx.x % HEADS, w = threadIdx.x / 32, lane = threadIdx.x % 32;
  const size_t row = seq < batch && orow ? (size_t)orow[seq] : (size_t)seq;
  uint2 *dst = reinterpret_cast<uint2 *>(out + row * ostride + (size_t)h * HEAD_DIM) + lane;
  if (seq >= batch) {
    if (w == 0) *dst = make_uint2(0u, 0u);
    return;
  }
  const float *P = part + (size_t)seq * nparts * PART;
  float m = -INFINITY, l = 0.f;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s = w; s < nparts; s += CW) {
    const float *ps = P + (size_t)s * PART;
    const float ms = ps[HEADS * HEAD_DIM + h], ls = ps[HEADS * HEAD_DIM + HEADS + h];
    const float4 o = reinterpret_cast<const float4 *>(ps + h * HEAD_DIM)[lane];
    const float mn = fmaxf(m, ms), x = exp2f(m - mn), y = exp2f(ms - mn);
    l = l * x + ls * y;
    a = make_float4(a.x * x + o.x * y, a.y * x + o.y * y, a.z * x + o.z * y, a.w * x + o.w * y);
    m = mn;
  }
  sacc[w][lane] = a;
  if (lane == 0) { sm_[w] = m; sl[w] = l; }
  __syncthreads();
  if (w) return;
  float M = sm_[0];
#pragma unroll
  for (int v = 1; v < CW; ++v) M = fmaxf(M, sm_[v]);
  float L = 0.f;
  float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int v = 0; v < CW; ++v) {
    const float x = exp2f(sm_[v] - M);
    const float4 o = sacc[v][lane];
    L += sl[v] * x;
    A = make_float4(A.x + o.x * x, A.y + o.y * x, A.z + o.z * x, A.w + o.w * x);
  }
  float r0, r1;
  const float iz = 0.f;
  *dst = L > 0.f ? make_uint2(bf16x2_rn(A.x / L, A.y / L, r0, r1), bf16x2_rn(A.z / L, A.w / L, r0, r1))
                 : make_uint2(bf16x2_rn(iz, iz, r0, r1), bf16x2_rn(iz, iz, r0, r1));
}

// A warp's words, one a lane (dst nullptr: none), stored through shared memory a whole word to Q = NL/4 lanes, so a store
// instruction writes whole 32-byte sectors (a lane's own 16-byte stores at a word's stride would each write half of one).
// Every lane of the warp calls it; blocks are at most 32 NW threads.
constexpr int SQ = NL / 4;
static_assert(NL % 4 == 0 && 32 % SQ == 0, "a word is whole 16-byte chunks, a whole number of them to a warp");

template <int NW = 8>
__device__ __forceinline__ void store_warp(u32 *dst, const u32 (&x)[NL]) {
  __shared__ uint4 sw[NW][32][SQ + 1];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int q = 0; q < SQ; ++q) sw[w][lane][q] = make_uint4(x[4 * q], x[4 * q + 1], x[4 * q + 2], x[4 * q + 3]);
  __syncwarp();
  const int sub = lane % SQ, grp = lane / SQ;
#pragma unroll
  for (int r = 0; r < SQ; ++r) {
    const int o = r * (32 / SQ) + grp;
    uint4 *d = reinterpret_cast<uint4 *>(__shfl_sync(0xffffffffu, reinterpret_cast<unsigned long long>(dst), o));
    if (d) d[sub] = sw[w][o][sub];
  }
  __syncwarp();
}

// A step's new tokens into the cache: token i's head g, K then V, as HW words at slot slots[i] (vLLM's slot mapping,
// block bs slots; negative: padding, skipped), encoded at k rounds (ENC) or as they are, under the slot's bound offset
// bj + slot NL (nullptr: the seed's key), and with plain the slot's flag set to whether it went as it is. A thread a word.
template <bool ENC>
__global__ void __launch_bounds__(128) pg_append_kernel(const uint16_t *__restrict__ key, const uint16_t *__restrict__ val,
                                                        size_t kstride, size_t vstride, const long long *__restrict__ slots,
                                                        int n, u32 *__restrict__ cache, int bs, int sb, int sh, int ss,
                                                        u64 store_base, int layers, int layer, int k,
                                                        const u32 *__restrict__ bj, uint8_t *__restrict__ plain) {
  const u64 u = (u64)blockIdx.x * 128 + threadIdx.x;
  u32 x[NL] = {};
  u32 *dst = nullptr;
  if (u < (u64)n * KV_HEADS * HW) {
    const int j = (int)(u % HW), g = (int)(u / HW % KV_HEADS), i = (int)(u / ((u64)HW * KV_HEADS));
    const long long slot = slots[i];
    if (slot >= 0) {
      const uint16_t *src = (j < HW / 2 ? key + (size_t)i * kstride : val + (size_t)i * vstride) + (size_t)g * HEAD_DIM +
                            (size_t)(j % (HW / 2)) * (WB / 2);
      load(reinterpret_cast<const u32 *>(src), x);
      const u64 wl = (u64)(slot / bs) * sb + (u64)g * sh + (u64)(slot % bs) * ss + j;
      if constexpr (ENC)
        encode_word(x, store_base + ((u64)slot * layers + layer) * TLW + pg_wi(g, j), k, bj ? bj + slot * NL : nullptr);
      dst = cache + wl * NL;
      if (plain && g == 0 && j == 0) plain[slot] = !ENC;
    }
  }
  store_warp(dst, x);
}

// pg_append_kernel over layers: layer l's row i of rows at u = (l rows + i) KV_HEADS HW + g HW + j; its table entry
// (PG_TAB words): key, value, their row strides, slots, rows used, cache, store base, layer count, layer index, the
// layer's bound offsets (0: the seed's key) and its plaintext flags (0: none).
constexpr int PG_TAB = 12;
template <bool ENC>
__global__ void __launch_bounds__(128) pg_append_multi_kernel(const u64 *__restrict__ tab, int layers, int rows, int bs,
                                                              int sb, int sh, int ss, int k) {
  const u64 u = (u64)blockIdx.x * 128 + threadIdx.x;
  const u64 per = (u64)rows * KV_HEADS * HW;
  u32 x[NL] = {};
  u32 *dst = nullptr;
  if (u < (u64)layers * per) {
    const u64 *e = tab + PG_TAB * (u / per);
    const u64 r = u % per;
    const int j = (int)(r % HW), g = (int)(r / HW % KV_HEADS), i = (int)(r / ((u64)HW * KV_HEADS));
    const long long slot = i < (int)e[5] ? reinterpret_cast<const long long *>(e[4])[i] : -1;
    if (slot >= 0) {
      const uint16_t *src = (j < HW / 2 ? reinterpret_cast<const uint16_t *>(e[0]) + (size_t)i * e[2]
                                        : reinterpret_cast<const uint16_t *>(e[1]) + (size_t)i * e[3]) +
                            (size_t)g * HEAD_DIM + (size_t)(j % (HW / 2)) * (WB / 2);
      load(reinterpret_cast<const u32 *>(src), x);
      const u64 wl = (u64)(slot / bs) * sb + (u64)g * sh + (u64)(slot % bs) * ss + j;
      if constexpr (ENC) {
        const u32 *bj = reinterpret_cast<const u32 *>(e[10]);
        encode_word(x, e[7] + ((u64)slot * e[8] + e[9]) * TLW + pg_wi(g, j), k, bj ? bj + slot * NL : nullptr);
      }
      dst = reinterpret_cast<u32 *>(e[6]) + wl * NL;
      if (e[11] && g == 0 && j == 0) reinterpret_cast<uint8_t *>(e[11])[slot] = !ENC;
    }
  }
  store_warp(dst, x);
}

// Deferred residency's encode over layers, in place: layer l's row i of rows at u = (l rows + i) KV_HEADS HW + g HW + j;
// its table entry (PG_ENC words): slots, rows used, cache, store base, layer count, layer index, the layer's bound
// offsets (0: the seed's key) and its plaintext flags (0: none, every listed slot encoded). A slot whose flag is clear is
// already encoded and left as it is; pg_settle_kernel clears the flags after.
constexpr int PG_ENC = 8;
__global__ void __launch_bounds__(128) pg_encode_kernel(const u64 *__restrict__ tab, int layers, int rows, int bs, int sb,
                                                        int sh, int ss, int k) {
  const u64 u = (u64)blockIdx.x * 128 + threadIdx.x;
  const u64 per = (u64)rows * KV_HEADS * HW;
  u32 x[NL] = {};
  u32 *dst = nullptr;
  if (u < (u64)layers * per) {
    const u64 *e = tab + PG_ENC * (u / per);
    const u64 r = u % per;
    const int j = (int)(r % HW), g = (int)(r / HW % KV_HEADS), i = (int)(r / ((u64)HW * KV_HEADS));
    const long long slot = i < (int)e[1] ? reinterpret_cast<const long long *>(e[0])[i] : -1;
    if (slot >= 0 && (!e[7] || reinterpret_cast<const uint8_t *>(e[7])[slot])) {
      u32 *w = reinterpret_cast<u32 *>(e[2]) + ((u64)(slot / bs) * sb + (u64)g * sh + (u64)(slot % bs) * ss + j) * NL;
      load(w, x);
      const u32 *bj = reinterpret_cast<const u32 *>(e[6]);
      encode_word(x, e[3] + ((u64)slot * e[4] + e[5]) * TLW + pg_wi(g, j), k, bj ? bj + slot * NL : nullptr);
      dst = w;
    }
  }
  store_warp(dst, x);
}

__global__ void __launch_bounds__(128) pg_settle_kernel(const u64 *__restrict__ tab, int layers, int rows) {
  const u64 u = (u64)blockIdx.x * 128 + threadIdx.x;
  if (u >= (u64)layers * rows) return;
  const u64 *e = tab + PG_ENC * (u / rows);
  const int i = (int)(u % rows);
  const long long slot = i < (int)e[1] ? reinterpret_cast<const long long *>(e[0])[i] : -1;
  if (e[7] && slot >= 0) reinterpret_cast<uint8_t *>(e[7])[slot] = 0;
}

// The lifetime cap's re-key in place: layer l's row i of rows at u = (l rows + i) KV_HEADS HW + g HW + j; its table
// entry (PG_REKEY words): slots, rows used, cache, store base, layer count, layer index, the layer's bound offsets
// ([physical slot][NL], a listed slot's its old b_j; 0: the layer is under the seed's key and left as it is), the new
// offsets b_j'(c_a') ([row][NL]) and the plaintext flags (0: none). A listed slot's words are decoded under the old
// offset and encoded under the new one in place unless its flag is set (plaintext, left as it is with its flag). The
// CTAs stride over the words, so a grid of fewer than every SM's resident CTAs leaves the rest of each SM to serving;
// pg_rekey_settle_kernel writes the new offsets over the old after.
constexpr int PG_REKEY = 9;
__global__ void __launch_bounds__(128) pg_rekey_kernel(const u64 *__restrict__ tab, int layers, int rows, int bs, int sb,
                                                       int sh, int ss, int k) {
  const u64 per = (u64)rows * KV_HEADS * HW, words = (u64)layers * per;
  for (u64 u0 = (u64)blockIdx.x * 128; u0 < words; u0 += (u64)gridDim.x * 128) {
    const u64 u = u0 + threadIdx.x;
    u32 x[NL] = {};
    u32 *dst = nullptr;
    if (u < words) {
      const u64 *e = tab + PG_REKEY * (u / per);
      const u64 r = u % per;
      const int j = (int)(r % HW), g = (int)(r / HW % KV_HEADS), i = (int)(r / ((u64)HW * KV_HEADS));
      const long long slot = i < (int)e[1] ? reinterpret_cast<const long long *>(e[0])[i] : -1;
      const u32 *bj = reinterpret_cast<const u32 *>(e[6]);
      if (slot >= 0 && bj && !pg_plain(reinterpret_cast<const uint8_t *>(e[8]), slot)) {
        u32 *w = reinterpret_cast<u32 *>(e[2]) + ((u64)(slot / bs) * sb + (u64)g * sh + (u64)(slot % bs) * ss + j) * NL;
        load(w, x);
        const u64 at = e[3] + ((u64)slot * e[4] + e[5]) * TLW + pg_wi(g, j);
        decode_word(x, at, k, bj + slot * NL);
        encode_word(x, at, k, reinterpret_cast<const u32 *>(e[7]) + (u64)i * NL);
        dst = w;
      }
    }
    store_warp<4>(dst, x);
  }
}

// Each listed slot's new offset into the layer's bound offsets, its flag set or not, so a plaintext slot's later
// pg_encode takes its slot's current key. A thread a (layer, row, 16 bytes).
__global__ void __launch_bounds__(128) pg_rekey_settle_kernel(const u64 *__restrict__ tab, int layers, int rows) {
  constexpr int Q = NL / 4;
  const u64 u = (u64)blockIdx.x * 128 + threadIdx.x;
  if (u >= (u64)layers * rows * Q) return;
  const u64 *e = tab + PG_REKEY * (u / ((u64)rows * Q));
  const int i = (int)(u / Q % rows), q = (int)(u % Q);
  const long long slot = i < (int)e[1] ? reinterpret_cast<const long long *>(e[0])[i] : -1;
  if (e[6] && slot >= 0)
    reinterpret_cast<uint4 *>(e[6])[slot * Q + q] = reinterpret_cast<const uint4 *>(e[7])[(u64)i * Q + q];
}

// nb whole blocks (blocks[b], bw words each, contiguous) of a layer's cache into out, block b at word b bw: decoded
// (ENC, under each slot's bound offset in bj, nullptr: the seed's key; a plaintext slot copied) or copied. A thread a
// word.
template <bool ENC>
__global__ void __launch_bounds__(128) pg_gather_kernel(const u32 *__restrict__ cache, const int *__restrict__ blocks,
                                                        int nb, int bw, int bs, int sh, int ss, u32 *__restrict__ out,
                                                        u64 store_base, int layers, int layer, int k,
                                                        const u32 *__restrict__ bj, const uint8_t *__restrict__ plain) {
  const u64 u = (u64)blockIdx.x * 128 + threadIdx.x;
  u32 x[NL] = {};
  u32 *dst = nullptr;
  if (u < (u64)nb * bw) {
    const unsigned b = (unsigned)u / (unsigned)bw, w = (unsigned)u % (unsigned)bw;   // pg_gather: nb bw < 2^32
    const u64 wl = (u64)blocks[b] * bw + w;
    load(cache + wl * NL, x);
    const int g = sh < ss ? (w % ss) / sh : w / sh;
    const int so = sh < ss ? w / ss : (w % sh) / ss;
    const int j = w - g * sh - so * ss;
    const u64 slot = (u64)blocks[b] * bs + so;
    if constexpr (ENC)
      if (!pg_plain(plain, slot))
        decode_word(x, store_base + (slot * layers + layer) * TLW + pg_wi(g, j), k, bj ? bj + slot * NL : nullptr);
    dst = out + ((u64)blockIdx.x * 128 + threadIdx.x) * NL;   // u again, not held across the decode (ptxas spills it)
  }
  store_warp(dst, x);
}

// Receipt-capable gather: one warp per KV head and one CTA per eight token slots. It decodes and copies each word as
// pg_gather does, and feeds that same encoded word to form 13's IMMA sketch before it leaves registers (a plaintext
// slot's is not marked used, so it gets no receipt).
template <bool ENC>
__global__ void __launch_bounds__(32 * KV_HEADS) pg_gather_rc_kernel(
    const u32 *__restrict__ cache, const int *__restrict__ blocks, int nb, int bs, int sb, int sh, int ss,
    u32 *__restrict__ out, u64 store_base, int layers, int layer, int k, int *__restrict__ rc_sk,
    int *__restrict__ used, const int *__restrict__ epoch, const u32 *__restrict__ bj,
    const uint8_t *__restrict__ plain) {
  const int g = threadIdx.x >> 5, lane = threadIdx.x & 31, t = lane >> 2, j = lane & 3;
  const u64 q = (u64)blockIdx.x * 8 + t;
  u32 x[NL] = {};
  u32 *dst = nullptr;
  if (q < (u64)nb * bs) {
    const int bo = q / bs, so = q % bs, b = blocks[bo];
    const u64 slot = (u64)b * bs + so;
    const u64 wl = (u64)b * sb + (u64)g * sh + (u64)so * ss + j;
    load(cache + wl * NL, x);
    int c[4] = {};
#pragma unroll
    for (int kk = 0; kk < RK; ++kk) {
      const uint4 v = kRcA[g][kk][lane];
      imma16832(c, v.x, v.y, v.z, v.w, x[2 * kk], x[2 * kk + 1]);
    }
    const int gid = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int n = 2 * t4 + (e & 1);
      if (n < 8) {
        const u64 su = (u64)b * bs + ((u64)blockIdx.x * 8 + n) % bs;
        rc_sk[((su * KV_HEADS + g) * 16) + gid + 8 * (e >> 1)] = c[e];
      }
    }
    const bool pt = pg_plain(plain, slot);
    if (g == 0 && j == 0 && !pt) atomicExch(used + slot, *epoch);
    if constexpr (ENC)
      if (!pt) decode_word(x, store_base + (slot * layers + layer) * TLW + pg_wi(g, j), k, bj ? bj + slot * NL : nullptr);
    const u64 ow = (u64)bo * sb + (u64)g * sh + (u64)so * ss + j;
    dst = out + ow * NL;
  }
  store_warp(dst, x);
}

// wo[0] = 0 and wo[r + 1] = wo[r] + lens[r] (a negative length counted 0): one CTA of 1024 threads, 1024 requests a pass.
__global__ void __launch_bounds__(1024) pg_window_offsets_kernel(const int *__restrict__ lens, int batch,
                                                                 int *__restrict__ wo) {
  __shared__ int ws[32];
  __shared__ int carry;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) carry = 0;
  for (int b0 = 0; b0 < batch; b0 += 1024) {
    const int i = b0 + threadIdx.x;
    int x = i < batch ? max(lens[i], 0) : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, x, o);
      if (lane >= o) x += y;
    }
    if (lane == 31) ws[w] = x;
    __syncthreads();
    if (w == 0) {
      int v = ws[lane];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += y;
      }
      ws[lane] = v;
    }
    __syncthreads();
    const int at = carry + x + (w ? ws[w - 1] : 0);
    if (i < batch) wo[i + 1] = at;
    __syncthreads();
    if (threadIdx.x == 1023) carry = at;
    __syncthreads();
  }
  if (threadIdx.x == 0) wo[0] = 0;
}

// pg_window's words, a thread a word and the CTAs striding over the rows: row R's head g, word j (K's HW/2 then V's) of
// request r, the last r with wo[r] <= R, at token t = R − wo[r] when t < lens[r].
template <bool ENC>
__global__ void __launch_bounds__(128) pg_window_kernel(const u32 *__restrict__ cache, Pg pg, int order,
                                                        const int *__restrict__ wo, int batch, u64 cap,
                                                        uint16_t *__restrict__ outk, uint16_t *__restrict__ outv,
                                                        u64 ots, u64 ogs, int k) {
  constexpr int RW = KV_HEADS * HW;
  const u64 words = min((u64)max(wo[batch], 0), cap) * RW;
  for (u64 u0 = (u64)blockIdx.x * 128; u0 < words; u0 += (u64)gridDim.x * 128) {
    const u64 u = u0 + threadIdx.x;
    u32 x[NL] = {};
    u32 *dst = nullptr;
    if (u < words) {
      const u64 R = u / RW;
      const int g = (int)(u / HW % KV_HEADS), j = (int)(u % HW);
      int lo = 0, hi = batch;
      while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if ((u64)wo[mid] <= R) lo = mid;
        else hi = mid;
      }
      const long long t = (long long)R - wo[lo];
      const int len = pg.lens[lo];
      if (t >= 0 && t < len) {
        if (pg.ck && t == len - 1) {
          const size_t crow = pg.qrow ? (size_t)pg.qrow[lo] : (size_t)lo;
          load(reinterpret_cast<const u32 *>((j < HW / 2 ? pg.ck + crow * pg.cks : pg.cv + crow * pg.cvs) +
                                             (size_t)g * HEAD_DIM + (size_t)(j % (HW / 2)) * (WB / 2)), x);
        } else {
          const int *btr = pg.bt + (size_t)lo * pg.bstride;
          load(cache + pg_word(pg, btr, (int)t, g, j) * NL, x);
          if constexpr (ENC) {
            const u64 slot = pg_slot(pg, btr, (int)t);
            if (!pg_plain(pg.plain, slot))
              decode_word(x, pg.store_base + (slot * pg.layers + pg.layer) * TLW + (order ? g * HW + j : pg_wi(g, j)),
                          k, pg.bj ? pg.bj + slot * NL : nullptr);
          }
        }
        dst = reinterpret_cast<u32 *>((j < HW / 2 ? outk : outv) + R * ots + (u64)g * ogs +
                                      (u64)(j % (HW / 2)) * (WB / 2));
      }
    }
    store_warp<4>(dst, x);
  }
}

__global__ void __launch_bounds__(256) pg_reduce_kernel(const float *__restrict__ part, int nch, int n, int mn,
                                                        const float *__restrict__ xs, int xs_rows,
                                                        const float *__restrict__ ws, __nv_bfloat16 *__restrict__ out) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= mn) return;
  float4 a = *reinterpret_cast<const float4 *>(part + i);
  for (int c = 1; c < nch; ++c) {
    const float4 v = *reinterpret_cast<const float4 *>(part + (size_t)c * mn + i);
    a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
  }
  const float sc = xs[xs_rows ? i / n : 0] * ws[0];
  *reinterpret_cast<__nv_bfloat162 *>(out + i) = __floats2bfloat162_rn(a.x * sc, a.y * sc);
  *reinterpret_cast<__nv_bfloat162 *>(out + i + 2) = __floats2bfloat162_rn(a.z * sc, a.w * sc);
}

// A timing span's mark (step_spans.py, whose `account` is its reference): acc is [the stream's last mark (ns, 0:
// none), the current kind, then each kind's parts' ns and marks]. One thread, ordered by its stream alone.
__global__ void pg_span_kernel(unsigned long long *acc, int part, int kind, int parts) {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  if (kind >= 0) acc[1] = (unsigned long long)kind;
  if (acc[0]) {
    unsigned long long *k = acc + 2 + acc[1] * 2 * parts;
    k[part] += t - acc[0];
    k[parts + part] += 1;
  }
  acc[0] = t;
}

}  // namespace kv

extern "C" {

int kv_fold6() { return kv::FOLD6; }

// Gate-only form-13 pass setup from the driver's 128-byte receipt prefix on the host. Serving uses kv_rc_pass with
// wt_pass's device midstate, so no seed or prefix crosses the host on the measured path.
int pg_rc_gate_pass(const uint32_t *bits, const uint8_t *prefix, cudaStream_t s) {
  if (!bits || !prefix || !kv::RC3_OK) return -1;
  uint64_t h[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                   0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  uint64_t w[16];
  for (int i = 0; i < 16; ++i) {
    w[i] = 0;
    for (int b = 0; b < 8; ++b) w[i] = w[i] << 8 | prefix[8 * i + b];
  }
  kv::sha512_compress(h, w);
  kv::rc_a_kernel<<<kv::KV_HEADS * kv::RK * 32 / 128, 128, 0, s>>>(reinterpret_cast<const uint8_t *>(bits));
  if (cudaMemcpyToSymbolAsync(kv::kKvMid, h, sizeof h, 0, cudaMemcpyHostToDevice, s) != cudaSuccess) return launched();
  return launched();
}

// heads, KV heads, head dim, words a (slot, head), bytes a word, tokens a tile, floats a partial, MMA warps a CTA,
// whether the field's words tile a head, the attention's shared memory.
void pg_shape(int *out) {
  out[0] = kv::HEADS; out[1] = kv::KV_HEADS; out[2] = kv::HEAD_DIM; out[3] = kv::HW; out[4] = kv::WB;
  out[5] = kv::MT; out[6] = kv::PART; out[7] = kv::MW; out[8] = kv::MMA_OK; out[9] = kv::pg_smem();
}

// One layer's decode attention for batch sequences (lens, block table bt) into nrows rows of out (bf16, row stride
// ostride; a sequence's row qrow[i], or i with qrow null, rows past batch zeroed), q's row likewise (row stride
// qstride); part holds batch splits MW partials (pg_shape's out[6] floats each). encoded = 0 plain, 1 its words at k
// rounds (the same output). ck, cv (row strides cks, cvs): each sequence's last token's K and V, at its row of q (null:
// from the pages). bj (encoded only): [physical slot][NL] the layer's bound offsets (null: the seed's key). plain
// (encoded only): [physical slot] the layer's plaintext flags (null: every slot encoded).
int pg_attn(const void *kv_, int encoded, const uint16_t *q, size_t qstride, float qscale, uint16_t *out,
            size_t ostride, int nrows, float *part, const int *lens, const int *bt, const int *qrow, int bstride,
            int bs, int sb, int sh, int ss, unsigned long long store_base, int layers, int layer,
            int batch, int splits, int k, const uint16_t *ck, const uint16_t *cv, size_t cks, size_t cvs,
            int *rc_sk, int *used, const int *epoch, const uint32_t *bj, const uint8_t *plain, cudaStream_t s) {
  if (!kv::MMA_OK) return -2;
  const bool rc = rc_sk || used || epoch;
  if (batch < 0 || nrows < batch || splits < 1 || bs < 1 || (encoded && (k < 1 || k > kv::KMAX)) || encoded < 0 ||
      encoded > 1 || (qrow && nrows != batch) || (!ck != !cv) || layers < 1 || layer < 0 || layer >= layers ||
      (rc && (!encoded || !rc_sk || !used || !epoch)) || ((bj || plain) && !encoded))
    return -1;
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(kv::pg_attn_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::pg_smem());
    cudaFuncSetAttribute(kv::pg_attn_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::pg_smem());
    cudaFuncSetAttribute(kv::pg_attn_kernel<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::pg_rc_smem());
    set = true;
  }
  const kv::Pg pg{lens, bt, qrow, bstride, bs, sb, sh, ss, store_base, layers, layer, ck, cv, cks, cvs,
                  rc_sk, used, epoch, bj, plain};
  if (batch) {
    const int grid = batch * splits * kv::KV_HEADS;
    if (rc)
      kv::pg_attn_kernel<true, true><<<grid, kv::MWT, kv::pg_rc_smem(), s>>>(
          static_cast<const uint8_t *>(kv_), q, qstride, qscale, part, pg, splits, k);
    else if (encoded)
      kv::pg_attn_kernel<true, false><<<grid, kv::MWT, kv::pg_smem(), s>>>(
          static_cast<const uint8_t *>(kv_), q, qstride, qscale, part, pg, splits, k);
    else
      kv::pg_attn_kernel<false, false><<<grid, kv::MWT, kv::pg_smem(), s>>>(
          static_cast<const uint8_t *>(kv_), q, qstride, qscale, part, pg, splits, k);
  }
  if (nrows)
    kv::pg_combine<<<nrows * kv::HEADS, 32 * kv::CW, 0, s>>>(part, out, ostride, qrow, batch, splits * kv::MW);
  return launched();
}

// bj (encoded only): [physical slot][NL] the layer's bound offsets, each appended token-layer encoded under its slot's
// (null: the seed's key). plain: [physical slot] the layer's plaintext flags, each appended slot's set to !encoded (an
// encoded cache's deferred append is encoded = 0 with plain; pg_encode then encodes it in place).
int pg_append(const uint16_t *key, const uint16_t *val, size_t kstride, size_t vstride, const long long *slots, int n,
              void *cache, int bs, int sb, int sh, int ss, unsigned long long store_base, int layers, int layer,
              int encoded, int k, const uint32_t *bj, uint8_t *plain, cudaStream_t s) {
  if (n < 0 || bs < 1 || layers < 1 || layer < 0 || layer >= layers ||
      (encoded && (k < 1 || k > kv::KMAX)) || (bj && !encoded) || !kv::MMA_OK) return -1;
  if (!n) return 0;
  const unsigned long long words = (unsigned long long)n * kv::KV_HEADS * kv::HW;
  const unsigned grid = (unsigned)((words + 127) / 128);
  (encoded ? kv::pg_append_kernel<true> : kv::pg_append_kernel<false>)<<<grid, 128, 0, s>>>(
      key, val, kstride, vstride, slots, n, static_cast<uint32_t *>(cache), bs, sb, sh, ss, store_base, layers, layer, k,
      bj, plain);
  return launched();
}

// layers layers' appends in one launch: tab (device) holds each layer's PG_TAB words (pg_append_multi_kernel), rows its
// most rows; every layer's cache in one layout (bs, sb, sh, ss).
int pg_append_multi(const unsigned long long *tab, int layers, int rows, int bs, int sb, int sh, int ss, int encoded,
                    int k, cudaStream_t s) {
  if (layers < 0 || rows < 0 || bs < 1 || (encoded && (k < 1 || k > kv::KMAX)) || !kv::MMA_OK) return -1;
  if (!layers || !rows) return 0;
  const unsigned long long words = (unsigned long long)layers * rows * kv::KV_HEADS * kv::HW;
  (encoded ? kv::pg_append_multi_kernel<true> : kv::pg_append_multi_kernel<false>)<<<(unsigned)((words + 127) / 128),
                                                                                       128, 0, s>>>(
      reinterpret_cast<const kv::u64 *>(tab), layers, rows, bs, sb, sh, ss, k);
  return launched();
}

// Deferred residency's encode, once the append's coin has arrived: layers layers' listed slots encoded in place at k
// rounds under their bound offsets b_j(c_a), and their flags cleared. tab (device) holds each layer's PG_ENC words
// (pg_encode_kernel), rows its most rows; every layer's cache in one layout. A layer's listed slots are distinct, and
// nothing reads those layers until it is done (the stack orders it before the next step's reads).
int pg_encode(const unsigned long long *tab, int layers, int rows, int bs, int sb, int sh, int ss, int k,
              cudaStream_t s) {
  if (layers < 0 || rows < 0 || bs < 1 || k < 1 || k > kv::KMAX || !kv::MMA_OK) return -1;
  if (!layers || !rows) return 0;
  const auto *t = reinterpret_cast<const kv::u64 *>(tab);
  const unsigned long long words = (unsigned long long)layers * rows * kv::KV_HEADS * kv::HW;
  kv::pg_encode_kernel<<<(unsigned)((words + 127) / 128), 128, 0, s>>>(t, layers, rows, bs, sb, sh, ss, k);
  kv::pg_settle_kernel<<<(unsigned)(((unsigned long long)layers * rows + 127) / 128), 128, 0, s>>>(t, layers, rows);
  return launched();
}

// Every SM's resident CTAs of pg_rekey_kernel.
int pg_rekey_ctas() {
  int dev = 0, sms = 0, per = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kv::pg_rekey_kernel, 128, 0);
  return std::max(1, sms * per);
}

// The lifetime cap's re-key (remote_bandwidth's PROTOCOL.md §4: a block kept past L windows is appended again at a
// fresh index under a fresh coin, in its slot), in place: layers layers' listed slots decoded under their old bound
// offsets and encoded under their new ones at k rounds, then the new offsets written over the old; a plaintext slot's
// words and flag are left as they are, its offset moved too. tab (device) holds each layer's PG_REKEY words
// (pg_rekey_kernel), rows its most rows; every layer's cache in one layout. A layer's listed slots are distinct, and
// nothing reads them until it is done: the caller orders it after their last read under the old offsets and before
// their first under the new. It reads and writes nothing else. ctas is the grid: 0 a word a thread (CTAs that end after
// one word each), a count that many CTAs striding over the words. A low-priority stream preempts no CTA, and the room a
// finished CTA frees seldom fits a serving kernel's, so beside serving a caller caps the grid and serving keeps the
// room the cap leaves (pg_rekey_ctas(), every SM's resident CTAs, leaves none).
int pg_rekey(const unsigned long long *tab, int layers, int rows, int bs, int sb, int sh, int ss, int k, int ctas,
             cudaStream_t s) {
  if (layers < 0 || rows < 0 || bs < 1 || k < 1 || k > kv::KMAX || ctas < 0 || !kv::MMA_OK || (layers && rows && !tab))
    return -1;
  if (!layers || !rows) return 0;
  const auto *t = reinterpret_cast<const kv::u64 *>(tab);
  const unsigned long long words = (unsigned long long)layers * rows * kv::KV_HEADS * kv::HW;
  const unsigned grid = ctas ? (unsigned)ctas : (unsigned)((words + 127) / 128);
  kv::pg_rekey_kernel<<<grid, 128, 0, s>>>(t, layers, rows, bs, sb, sh, ss, k);
  const unsigned long long limbs = (unsigned long long)layers * rows * (kv::NL / 4);
  kv::pg_rekey_settle_kernel<<<(unsigned)((limbs + 127) / 128), 128, 0, s>>>(t, layers, rows);
  return launched();
}

// bj (encoded only): [physical slot][NL] the layer's bound offsets (null: the seed's key). plain (encoded only):
// [physical slot] the layer's plaintext flags (null: every slot encoded).
int pg_gather(const void *cache, const int *blocks, int nb, int bw, int bs, int sb, int sh, int ss, void *out,
              unsigned long long store_base, int layers, int layer, int encoded, int k, int *rc_sk, int *used,
              const int *epoch, const uint32_t *bj, const uint8_t *plain, cudaStream_t s) {
  const bool rc = rc_sk || used || epoch;
  if (nb < 0 || bw < 1 || bs < 1 || layers < 1 || layer < 0 || layer >= layers ||
      (unsigned long long)nb * bw >= 1ull << 32 || (encoded && (k < 1 || k > kv::KMAX)) ||
      (rc && (!encoded || !rc_sk || !used || !epoch)) || ((bj || plain) && !encoded)) return -1;
  if (!nb) return 0;
  if (rc) {
    kv::pg_gather_rc_kernel<true><<<(unsigned)(((unsigned long long)nb * bs + 7) / 8), 32 * kv::KV_HEADS, 0, s>>>(
        static_cast<const uint32_t *>(cache), blocks, nb, bs, sb, sh, ss, static_cast<uint32_t *>(out), store_base,
        layers, layer, k, rc_sk, used, epoch, bj, plain);
  } else {
    const unsigned long long words = (unsigned long long)nb * bw;
    if (encoded)
      kv::pg_gather_kernel<true><<<(unsigned)((words + 127) / 128), 128, 0, s>>>(
          static_cast<const uint32_t *>(cache), blocks, nb, bw, bs, sh, ss, static_cast<uint32_t *>(out), store_base,
          layers, layer, k, bj, plain);
    else
      kv::pg_gather_kernel<false><<<(unsigned)((words + 127) / 128), 128, 0, s>>>(
          static_cast<const uint32_t *>(cache), blocks, nb, bw, bs, sh, ss, static_cast<uint32_t *>(out), store_base,
          layers, layer, k, nullptr, nullptr);
  }
  return launched();
}

// pg_window's offsets on the device: wo[0] = 0 and wo[r + 1] = wo[r] + lens[r] (a negative length counted 0), so
// request r's window is rows wo[r] … wo[r + 1] − 1 of the working buffer and wo[batch] its rows. A step's layers share it.
int pg_window_offsets(const int *lens, int batch, int *wo, cudaStream_t s) {
  if (batch < 0 || !wo || (batch && !lens)) return -1;
  kv::pg_window_offsets_kernel<<<1, 1024, 0, s>>>(lens, batch, wo);
  return launched();
}

// pg_window's default grid: every SM's resident CTAs of the copy (encoded 0) or the decode (1).
int pg_window_ctas(int encoded) {
  int dev = 0, sms = 0, per = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, encoded ? kv::pg_window_kernel<true> : kv::pg_window_kernel<false>,
                                                128, 0);
  return std::max(1, sms * per);
}

// One layer's decode windows for batch requests (lens, block table bt of row stride bstride) into outk and outv (bf16):
// request r's token t at row wo[r] + t, wo nondecreasing from 0 with wo[r + 1] − wo[r] >= lens[r] (pg_window_offsets'
// or with gaps, left as they are), rows from cap on not written; head g of row R at R ots + g ogs elements, 16-byte
// aligned. encoded 0 copies; 1 decodes at k rounds under bj ([physical slot][NL], null: the seed's key), a slot that
// plain ([physical slot], null: none) flags read as it is. order: the codec's word order (the header). ck, cv (row strides
// cks, cvs; null: from the pages): each request's last token, at row qrow[r] (null: r). ctas: the grid (0: every SM's
// resident CTAs), fixed, so a CUDA graph replays the launch at any lengths.
int pg_window(const void *cache, int encoded, int order, const int *lens, const int *bt, int bstride, int bs, int sb,
              int sh, int ss, unsigned long long store_base, int layers, int layer, int k, const uint32_t *bj,
              const uint8_t *plain, const uint16_t *ck, const uint16_t *cv, size_t cks, size_t cvs, const int *qrow,
              const int *wo, int batch, unsigned long long cap, uint16_t *outk, uint16_t *outv,
              unsigned long long ots, unsigned long long ogs, int ctas, cudaStream_t s) {
  const auto misaligned = [](const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) != 0; };
  if (batch < 0 || bs < 1 || layers < 1 || layer < 0 || layer >= layers || encoded < 0 || encoded > 1 || order < 0 ||
      order > 1 || (encoded && (k < 1 || k > kv::KMAX)) || ((bj || plain) && !encoded) || (!ck != !cv) || ctas < 0 ||
      !kv::MMA_OK || (batch && (!cache || !lens || !bt || !wo || !outk || !outv)) || misaligned(cache) ||
      misaligned(outk) || misaligned(outv) || ots % 8 || ogs % 8 || misaligned(bj) ||
      (ck && (misaligned(ck) || misaligned(cv) || cks % 8 || cvs % 8)))
    return -1;
  if (!batch) return 0;
  static int resident[2] = {0, 0};
  if (!ctas) {
    if (!resident[encoded]) resident[encoded] = pg_window_ctas(encoded);
    ctas = resident[encoded];
  }
  const kv::Pg pg{lens, bt, qrow, bstride, bs, sb, sh, ss, store_base, layers, layer, ck, cv, cks, cvs,
                  nullptr, nullptr, nullptr, bj, plain};
  (encoded ? kv::pg_window_kernel<true> : kv::pg_window_kernel<false>)<<<ctas, 128, 0, s>>>(
      static_cast<const uint32_t *>(cache), pg, order, wo, batch, cap, outk, outv, ots, ogs, k);
  return launched();
}

int pg_receipts_sparse(const int *sk, uint32_t *tags, const int *used, const int *epoch,
                       unsigned long long slots, int layers, int layer, unsigned long long jbase, cudaStream_t s) {
  if (!sk || !tags || !used || !epoch || !slots || layers < 1 || layer < 0 || layer >= layers) return -1;
  kv::pg_receipt_sparse_kernel<<<(unsigned)((slots + 127) / 128), 128, 0, s>>>(
      sk, tags, used, epoch, slots, layers, layer, jbase);
  return launched();
}

int pg_reduce(const float *part, int nch, int M, int N, const float *xs, int xs_rows, const float *ws, void *out,
              cudaStream_t s) {
  if (M < 1 || N % 4 || nch < 1) return -1;
  const int mn = M * N;
  kv::pg_reduce_kernel<<<(mn / 4 + 255) / 256, 256, 0, s>>>(part, nch, N, mn, xs, xs_rows, ws,
                                                             static_cast<__nv_bfloat16 *>(out));
  return launched();
}

// A span mark on s: the time since s's last mark into part `part` (of `parts`) of the current kind, set to `kind`
// first when kind >= 0 (of `kinds`).
int pg_span(unsigned long long *acc, int part, int kind, int parts, int kinds, cudaStream_t s) {
  if (!acc || parts < 1 || part < 0 || part >= parts || kinds < 1 || kind >= kinds) return -1;
  kv::pg_span_kernel<<<1, 1, 0, s>>>(acc, part, kind, parts);
  return launched();
}
}
