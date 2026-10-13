// The HBM-exhaustion decode step's KV library (`step.py`): `benchmarks/word_codec/kv.cu` (its codec, read and
// attention) and the append, so one build holds the KV key. Build with -DKV_FIELD (and KV_MASK, KV_FOLD6) at wt.cu's
// field and codec (step.sh).
//
// kv_append writes `batch` sequences' new token at slot t, layers l0 .. l0 + nl - 1, into the cache in kv_attn's layout:
// encoded at k rounds (word w of token-layer (b ctx + t) layers + l at index (((b ctx + t) layers + l) TLW + w)), or
// plain at k = 0. `in` holds the tokens' plain words, [batch][nl][TLW] of NL limbs.
#include "../word_codec/kv.cu"

namespace kv {

constexpr int APPEND_THREADS = 64;
static_assert(TLW % 32 == 0, "a warp's words are one run of its token-layer");

// A warp's 32 words are consecutive in the cache, so it stages them in shared memory and stores them as whole rows: a
// word a lane would leave each of its 16-byte stores half a 32-byte sector.
__global__ void __launch_bounds__(APPEND_THREADS) append_kernel(const u32 *__restrict__ in, u32 *__restrict__ cache,
                                                                int batch, int ctx, int layers, int l0, int nl, int t,
                                                                int k) {
  constexpr int Q = NL / 4;   // a word's 16-byte pieces
  __shared__ uint4 sm[APPEND_THREADS / 32][32 * Q];
  const u64 u = (u64)blockIdx.x * APPEND_THREADS + threadIdx.x;
  if (u >= (u64)batch * nl * TLW) return;   // whole warps: the words are whole token-layers
  const int lane = threadIdx.x & 31;
  const int w = (int)(u % TLW), l = l0 + (int)(u / TLW % nl), b = (int)(u / ((u64)TLW * nl));
  const u64 i = (((u64)b * ctx + t) * layers + l) * TLW + w;
  u32 x[NL];
  load(in + u * NL, x);
  if (k) encode_word(x, i, k);
  uint4 *s = sm[threadIdx.x / 32];
#pragma unroll
  for (int e = 0; e < Q; ++e) s[lane * Q + e] = make_uint4(x[4 * e], x[4 * e + 1], x[4 * e + 2], x[4 * e + 3]);
  __syncwarp();
  uint4 *g = reinterpret_cast<uint4 *>(cache + (i - lane) * NL);
#pragma unroll
  for (int e = 0; e < Q; ++e) g[lane + 32 * e] = s[lane + 32 * e];
}

}  // namespace kv

extern "C" int kv_fold6() { return kv::FOLD6; }

extern "C" int kv_append(const void *in, void *cache, int batch, int ctx, int layers, int l0, int nl, int t, int k,
                         cudaStream_t s) {
  if (kv::TLE != kv::TL_BYTES || batch < 1 || t < 0 || t >= ctx || l0 < 0 || nl < 1 || l0 + nl > layers || k < 0 ||
      k > kv::KMAX)
    return -1;
  const unsigned long long words = (unsigned long long)batch * nl * kv::TLW;
  kv::append_kernel<<<(unsigned)((words + kv::APPEND_THREADS - 1) / kv::APPEND_THREADS), kv::APPEND_THREADS, 0, s>>>(
      static_cast<const uint32_t *>(in), static_cast<uint32_t *>(cache), batch, ctx, layers, l0, nl, t, k);
  return launched();
}
