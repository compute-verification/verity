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

__global__ void __launch_bounds__(APPEND_THREADS) append_kernel(const u32 *__restrict__ in, u32 *__restrict__ cache,
                                                                int batch, int ctx, int layers, int l0, int nl, int t,
                                                                int k) {
  const u64 u = (u64)blockIdx.x * APPEND_THREADS + threadIdx.x;
  if (u >= (u64)batch * nl * TLW) return;
  const int w = (int)(u % TLW), l = l0 + (int)(u / TLW % nl), b = (int)(u / ((u64)TLW * nl));
  const u64 i = (((u64)b * ctx + t) * layers + l) * TLW + w;
  u32 x[NL];
  load(in + u * NL, x);
  if (k) encode_word(x, i, k);
  store(cache + i * NL, x);
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
