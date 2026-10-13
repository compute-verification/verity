// vLLM's paged KV cache in `benchmarks/word_codec/kv.cu`'s codec: decode attention over the pages, the append of a
// step's new tokens, and the gather of whole pages for prefill (FlashAttention reads them from a scratch). Build with
// -DKV_FIELD, -DKV_MASK, -DKV_FOLD6 and -DKV_QHEADS at the encoded stack's codec (run.sh).
//
// A layer's cache is vLLM's own tensor, [blocks][KV heads][slots][K ‖ V] or any order of the inner three whose block
// is contiguous: head g of slot s of block b, its 256 bytes of K then 256 of V, is HW words at word offset
// b sb + g sh + s ss (sb, sh, ss its strides in words), and its word j has index lbase + b sb + g sh + s ss + j
// (lbase the layer's first index, so no two words of the pool share one). A word's index is its place in the pool,
// so the append, the attention and the gather agree on it whatever the block table.
//
// pg_attn is kv.cu's warp-specialized tensor-core form (attn_mma_ws_kernel) over pages: MDW decoder warps, a lane a
// word of every tile, load the tile's words of one KV head from the pages (the block table), decode them (encoded) or
// pass them on (plain) into a ring of MNB tiles, and the MW MMA warps fold each tile in kv.cu's arithmetic
// (mma_chunk), so plain and encoded agree bit for bit. A sequence's tokens are split into `splits` ranges of whole
// tiles by its own length (lens), so the grid is fixed and a CUDA graph replays it at any lengths; an empty split
// leaves no tokens (max -1e30, sum 0), a sequence of length 0 a zero row. q is read from vLLM's query (bf16, any row
// stride) and scaled by qscale (log2(e) / sqrt(128)) and rounded to bf16 in place of kv.cu's pre-scaled q; the output
// is bf16, vLLM's.
#include "../word_codec/kv.cu"

namespace kv {

struct Pg {
  const int *lens;   // [batch] tokens a sequence
  const int *bt;     // [batch][bstride] block ids
  const int *qrow;   // [batch] its row of q and of out (nullptr: the sequence's index)
  int bstride, bs;   // the block table's row stride, slots a block
  int sb, sh, ss;    // words between blocks, heads, slots
  u64 lbase;         // the layer's first word index
};

__device__ __forceinline__ u64 pg_word(const Pg &pg, const int *btr, int p, int g, int j) {
  return (u64)btr[p / pg.bs] * pg.sb + (u64)g * pg.sh + (u64)(p % pg.bs) * pg.ss + j;
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

template <bool ENC>
__global__ void __launch_bounds__(MWT, 2) pg_attn_kernel(const uint8_t *__restrict__ kv, const uint16_t *__restrict__ q,
                                                         size_t qstride, float qscale, float *__restrict__ part, Pg pg,
                                                         int splits, int k) {
  if constexpr (MMA_OK) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint64_t *full = reinterpret_cast<uint64_t *>(sm + MNB * MT * MTS), *empty = full + MNB;
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
    __syncthreads();
    if (w >= MW) {
      const int d = threadIdx.x - 32 * MW, t = d / HW, j = d % HW;
      const int *btr = pg.bt + (size_t)seq * pg.bstride;
      const u32 *base = reinterpret_cast<const u32 *>(kv);
      const auto has = [&](int tile) { return tile < ntiles && t0 + tile * MT + t < t1; };
      u32 x[NL];
      u64 wl = has(0) ? pg_word(pg, btr, t0 + t, g, j) : 0;
      if (has(0)) ldg_word(base + wl * NL, x);
      for (int tile = 0; tile < ntiles; ++tile) {
        const bool cur = has(tile), next = has(tile + 1);
        const u64 wn = next ? pg_word(pg, btr, t0 + (tile + 1) * MT + t, g, j) : 0;
        if (next) prefetch_word_l2(base + wn * NL);
        if (!cur) {
#pragma unroll
          for (int l = 0; l < NL; ++l) x[l] = 0u;
        } else if constexpr (ENC) {
          u32 tw[NL];
          tweak(pg.lbase + wl, tw);
          decode_word_t<true>(x, tw, k);
        }
        if (tile >= MNB) mbar_wait(empty + tile % MNB, (tile / MNB - 1) & 1);
        store(reinterpret_cast<u32 *>(sm + (tile % MNB) * MT * MTS + t * MTS + j * WB), x);
        mbar_arrive(full + tile % MNB);
        if (next) ldg_word(base + wn * NL, x);
        wl = wn;
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

// A step's new tokens into the cache: token i's head g, K then V, as HW words at slot slots[i] (vLLM's slot mapping,
// block bs slots; negative: padding, skipped), encoded at k rounds (ENC) or as they are. A thread a word.
template <bool ENC>
__global__ void __launch_bounds__(128) pg_append_kernel(const uint16_t *__restrict__ key, const uint16_t *__restrict__ val,
                                                        size_t kstride, size_t vstride, const long long *__restrict__ slots,
                                                        int n, u32 *__restrict__ cache, int bs, int sb, int sh, int ss,
                                                        u64 lbase, int k) {
  const u64 u = (u64)blockIdx.x * 128 + threadIdx.x;
  if (u >= (u64)n * KV_HEADS * HW) return;
  const int j = (int)(u % HW), g = (int)(u / HW % KV_HEADS), i = (int)(u / ((u64)HW * KV_HEADS));
  const long long slot = slots[i];
  if (slot < 0) return;
  const uint16_t *src = (j < HW / 2 ? key + (size_t)i * kstride : val + (size_t)i * vstride) + (size_t)g * HEAD_DIM +
                        (size_t)(j % (HW / 2)) * (WB / 2);
  u32 x[NL];
  load(reinterpret_cast<const u32 *>(src), x);
  const u64 wl = (u64)(slot / bs) * sb + (u64)g * sh + (u64)(slot % bs) * ss + j;
  if constexpr (ENC) encode_word(x, lbase + wl, k);
  store(cache + wl * NL, x);
}

// nb whole blocks (blocks[b], bw words each, contiguous) of a layer's cache into out, block b at word b bw: decoded
// (ENC) or copied. A thread a word.
template <bool ENC>
__global__ void __launch_bounds__(128) pg_gather_kernel(const u32 *__restrict__ cache, const int *__restrict__ blocks,
                                                        int nb, int bw, u32 *__restrict__ out, u64 lbase, int k) {
  const u64 u = (u64)blockIdx.x * 128 + threadIdx.x;
  if (u >= (u64)nb * bw) return;
  const int b = (int)(u / bw), w = (int)(u % bw);
  const u64 wl = (u64)blocks[b] * bw + w;
  u32 x[NL];
  load(cache + wl * NL, x);
  if constexpr (ENC) decode_word(x, lbase + wl, k);
  store(out + u * NL, x);
}

}  // namespace kv

extern "C" {

int kv_fold6() { return kv::FOLD6; }

// heads, KV heads, head dim, words a (slot, head), bytes a word, tokens a tile, floats a partial, MMA warps a CTA,
// whether the field's words tile a head, the attention's shared memory.
void pg_shape(int *out) {
  out[0] = kv::HEADS; out[1] = kv::KV_HEADS; out[2] = kv::HEAD_DIM; out[3] = kv::HW; out[4] = kv::WB;
  out[5] = kv::MT; out[6] = kv::PART; out[7] = kv::MW; out[8] = kv::MMA_OK; out[9] = kv::pg_smem();
}

// One layer's decode attention for batch sequences (lens, block table bt) into nrows rows of out (bf16, row stride
// ostride; a sequence's row qrow[i], or i with qrow null, rows past batch zeroed), q's row likewise (row stride
// qstride); part holds batch splits MW partials (pg_shape's out[6] floats each). encoded = 0 plain, 1 its words at k
// rounds (the same output).
int pg_attn(const void *kv_, int encoded, const uint16_t *q, size_t qstride, float qscale, uint16_t *out,
            size_t ostride, int nrows, float *part, const int *lens, const int *bt, const int *qrow, int bstride,
            int bs, int sb, int sh, int ss, unsigned long long lbase, int batch, int splits, int k, cudaStream_t s) {
  if (!kv::MMA_OK) return -2;
  if (batch < 0 || nrows < batch || splits < 1 || bs < 1 || (encoded && (k < 1 || k > kv::KMAX)) || encoded < 0 ||
      encoded > 1 || (qrow && nrows != batch))
    return -1;
  static bool set = false;
  if (!set) {
    for (auto *f : {kv::pg_attn_kernel<true>, kv::pg_attn_kernel<false>})
      cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::pg_smem());
    set = true;
  }
  const kv::Pg pg{lens, bt, qrow, bstride, bs, sb, sh, ss, lbase};
  if (batch) {
    const int grid = batch * splits * kv::KV_HEADS;
    (encoded ? kv::pg_attn_kernel<true> : kv::pg_attn_kernel<false>)<<<grid, kv::MWT, kv::pg_smem(), s>>>(
        static_cast<const uint8_t *>(kv_), q, qstride, qscale, part, pg, splits, k);
  }
  if (nrows)
    kv::pg_combine<<<nrows * kv::HEADS, 32 * kv::CW, 0, s>>>(part, out, ostride, qrow, batch, splits * kv::MW);
  return launched();
}

int pg_append(const uint16_t *key, const uint16_t *val, size_t kstride, size_t vstride, const long long *slots, int n,
              void *cache, int bs, int sb, int sh, int ss, unsigned long long lbase, int encoded, int k,
              cudaStream_t s) {
  if (n < 0 || bs < 1 || (encoded && (k < 1 || k > kv::KMAX)) || !kv::MMA_OK) return -1;
  if (!n) return 0;
  const unsigned long long words = (unsigned long long)n * kv::KV_HEADS * kv::HW;
  const unsigned grid = (unsigned)((words + 127) / 128);
  (encoded ? kv::pg_append_kernel<true> : kv::pg_append_kernel<false>)<<<grid, 128, 0, s>>>(
      key, val, kstride, vstride, slots, n, static_cast<uint32_t *>(cache), bs, sb, sh, ss, lbase, k);
  return launched();
}

int pg_gather(const void *cache, const int *blocks, int nb, int bw, void *out, unsigned long long lbase, int encoded,
              int k, cudaStream_t s) {
  if (nb < 0 || bw < 1 || (encoded && (k < 1 || k > kv::KMAX))) return -1;
  if (!nb) return 0;
  const unsigned long long words = (unsigned long long)nb * bw;
  (encoded ? kv::pg_gather_kernel<true> : kv::pg_gather_kernel<false>)<<<(unsigned)((words + 127) / 128), 128, 0, s>>>(
      static_cast<const uint32_t *>(cache), blocks, nb, bw, static_cast<uint32_t *>(out), lbase, k);
  return launched();
}
}
