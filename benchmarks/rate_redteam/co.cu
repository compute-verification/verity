// The red team's co-residency probe (lane ra-rate-redteam). In every block, warps [0, b3w) run ro-blake3/v1's keyed
// compression (#1805's tag) and the other warps run the word codec's k-round encode (the adversary's rebuild) or decode
// (the honest prover's), all for the same number of SM clocks. The decode-free bound's (1 - tau) charges the tag and the
// rebuild to one resource: each kind's rate, divided by its own rate when it has every warp, would then sum to at most 1.
// A sum above 1 is spare the bound doesn't count.
//
//   co gate  WHAT K KEYFILE OUT   fixed steps on two blocks; every chain's final value to OUT for gate.py, the
//                                 compression chains also checked here against the host build of the same function
//   co sweep WHAT K KEYFILE OUT   b3w = 0 … 8 at the occupancy limit, three reps each, JSON to OUT
//
// WHAT: 1 encode, 0 decode. KEYFILE: gate.py's "k c_0 … c_{k-1} tau tstep", decimal. Build with -DRT_B3V=1 for the
// compression's V1 form (adds on the FMA pipe).
#include "../word_codec/wc.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace rt {

constexpr int UE = 4;  // codec chains a thread
constexpr int UB = 2;  // compression chains a thread
constexpr int THREADS = 256;

__host__ __device__ inline uint32_t b3_seed(uint64_t cid, int i) { return (uint32_t)cid * 0x9E3779B9u + i; }

// RT_B3V=1: the compression with ra-codec-tensor's V1 split (wt.cu): each add an IMAD by a one ptxas can't fold, so it
// issues on the FMA pipe, and the rotations by 16 and 8 as byte permutes. The gate still checks it against bw::b3_compress.
#ifndef RT_B3V
#define RT_B3V 0
#endif
__constant__ uint32_t kOne;

__device__ __forceinline__ uint32_t imad1(uint32_t a, uint32_t one, uint32_t c) {
  uint32_t d;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(one), "r"(c));
  return d;
}

#define RT_G1(a, b, c, d, x, y)                     \
  a = imad1(b, one, imad1(x, one, a));              \
  d = __byte_perm(d ^ a, 0, 0x1032);                \
  c = imad1(d, one, c);                             \
  b = __funnelshift_r(b ^ c, b ^ c, 12);            \
  a = imad1(b, one, imad1(y, one, a));              \
  d = __byte_perm(d ^ a, 0, 0x0321);                \
  c = imad1(d, one, c);                             \
  b = __funnelshift_r(b ^ c, b ^ c, 7)

__device__ __forceinline__ void b3_compress_v1(const uint32_t cv[8], uint32_t m[16], uint32_t counter, uint32_t len,
                                               uint32_t flags, uint32_t out[8]) {
  const uint32_t one = kOne;
  uint32_t v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
  uint32_t v8 = B3_IV0, v9 = B3_IV1, v10 = B3_IV2, v11 = B3_IV3, v12 = counter, v13 = 0, v14 = len, v15 = flags;
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    RT_G1(v0, v4, v8, v12, m[0], m[1]);
    RT_G1(v1, v5, v9, v13, m[2], m[3]);
    RT_G1(v2, v6, v10, v14, m[4], m[5]);
    RT_G1(v3, v7, v11, v15, m[6], m[7]);
    RT_G1(v0, v5, v10, v15, m[8], m[9]);
    RT_G1(v1, v6, v11, v12, m[10], m[11]);
    RT_G1(v2, v7, v8, v13, m[12], m[13]);
    RT_G1(v3, v4, v9, v14, m[14], m[15]);
    if (r < 6) {
      const uint32_t t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = t[i];
    }
  }
  out[0] = v0 ^ v8;
  out[1] = v1 ^ v9;
  out[2] = v2 ^ v10;
  out[3] = v3 ^ v11;
  out[4] = v4 ^ v12;
  out[5] = v5 ^ v13;
  out[6] = v6 ^ v14;
  out[7] = v7 ^ v15;
}
#undef RT_G1

__device__ __forceinline__ void b3_tag(const uint32_t cv[8], uint32_t m[16], uint32_t counter, uint32_t len, uint32_t flags,
                                       uint32_t out[8]) {
#if RT_B3V
  b3_compress_v1(cv, m, counter, len, flags, out);
#else
  bw::b3_compress(cv, m, counter, len, flags, out);
#endif
}

inline bw::Key3 b3_key() {
  bw::Key3 key;
  for (int i = 0; i < 8; ++i) key.k[i] = 0x01000193u * (i + 1);
  return key;
}

// cycles > 0: run for that many SM clocks; cycles < 0: exactly -cycles steps. counts[0] compressions, counts[1] words.
template <int WHAT, int K>
__global__ void __launch_bounds__(THREADS) co_kernel(int b3w, long long cycles, uint64_t tstep, bw::Key3 key,
                                                     unsigned long long* counts, uint64_t* out_e, uint32_t* out_b) {
  const int warp = threadIdx.x >> 5;
  const uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const bool fixed = cycles < 0;
  const long long steps = -cycles;
  const long long t0 = clock64();
  unsigned long long n = 0;
  if (warp < b3w) {
    uint32_t h[UB][8];
#pragma unroll
    for (int u = 0; u < UB; ++u)
#pragma unroll
      for (int i = 0; i < 8; ++i) h[u][i] = b3_seed(id * UB + u, i);
    for (long long s = 0; fixed ? s < steps : clock64() - t0 < cycles; ++s, ++n) {
#pragma unroll
      for (int u = 0; u < UB; ++u) {
        uint32_t m[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) m[i] = i < 8 ? h[u][i] : (uint32_t)s + i;
        b3_tag(key.k, m, 0, 64, bw::B3_KEYED | bw::B3_CHUNK_START, h[u]);
      }
    }
    atomicAdd(&counts[0], n * UB);
    if (out_b)
#pragma unroll
      for (int u = 0; u < UB; ++u)
#pragma unroll
        for (int i = 0; i < 8; ++i) out_b[(id * UB + u) * 8 + i] = h[u][i];
  } else {
    uint64_t y[UE], t[UE];
#pragma unroll
    for (int u = 0; u < UE; ++u) {
      y[u] = wc::canon((id * UE + u) * 0x9E3779B97F4A7C15ull & wc::P);
      t[u] = wc::mulmod(wc::kTau, id * UE + u);
    }
    for (long long s = 0; fixed ? s < steps : clock64() - t0 < cycles; ++s, ++n) {
#pragma unroll
      for (int u = 0; u < UE; ++u) {
        y[u] = WHAT ? wc::encode_word<K>(y[u], t[u]) : wc::decode_word<K>(y[u], ~t[u]);
        t[u] = wc::addmod(t[u], tstep);
      }
    }
    atomicAdd(&counts[1], n * UE);
    if (out_e)
#pragma unroll
      for (int u = 0; u < UE; ++u) out_e[id * UE + u] = y[u];
  }
}

struct Run {
  double ms, comp_per_s, words_per_s;
};

template <int WHAT, int K>
Run launch(int grid, int b3w, long long cycles, uint64_t tstep, uint64_t* out_e, uint32_t* out_b) {
  unsigned long long* counts;
  cudaMalloc(&counts, 2 * sizeof(unsigned long long));
  cudaMemset(counts, 0, 2 * sizeof(unsigned long long));
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  co_kernel<WHAT, K><<<grid, THREADS>>>(b3w, cycles, tstep, b3_key(), counts, out_e, out_b);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  unsigned long long c[2];
  cudaMemcpy(c, counts, sizeof c, cudaMemcpyDeviceToHost);
  cudaFree(counts);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return {ms, c[0] / (ms * 1e-3), c[1] / (ms * 1e-3)};
}

template <int WHAT, int K>
int gate(uint64_t tstep, const char* out) {
  const int grid = 2, b3w = 3, steps = 3;
  const size_t n = (size_t)grid * THREADS;
  uint64_t* de;
  uint32_t* db;
  cudaMalloc(&de, n * UE * sizeof(uint64_t));
  cudaMalloc(&db, n * UB * 8 * sizeof(uint32_t));
  launch<WHAT, K>(grid, b3w, -steps, tstep, de, db);
  std::vector<uint64_t> e(n * UE);
  std::vector<uint32_t> b(n * UB * 8);
  cudaMemcpy(e.data(), de, e.size() * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data(), db, b.size() * 4, cudaMemcpyDeviceToHost);
  if (cudaGetLastError()) return 3;
  const bw::Key3 key = b3_key();
  int bad = 0, nb = 0;
  FILE* f = fopen(out, "w");
  for (size_t id = 0; id < n; ++id) {
    const bool tagger = (id % THREADS) / 32 < (size_t)b3w;
    if (tagger) {
      for (int u = 0; u < UB; ++u) {
        uint32_t h[8];
        for (int i = 0; i < 8; ++i) h[i] = b3_seed(id * UB + u, i);
        for (int s = 0; s < steps; ++s) {
          uint32_t m[16];
          for (int i = 0; i < 16; ++i) m[i] = i < 8 ? h[i] : (uint32_t)s + i;
          bw::b3_compress(key.k, m, 0, 64, bw::B3_KEYED | bw::B3_CHUNK_START, h);
        }
        ++nb;
        for (int i = 0; i < 8; ++i) bad += h[i] != b[(id * UB + u) * 8 + i];
      }
    } else {
      for (int u = 0; u < UE; ++u) fprintf(f, "%llu %llu\n", (unsigned long long)(id * UE + u), (unsigned long long)e[id * UE + u]);
    }
  }
  fprintf(f, "steps %d\n", steps);
  fclose(f);
  printf("gate: %d compression chains checked on the host, %d words differ\n", nb, bad);
  cudaFree(de);
  cudaFree(db);
  return bad ? 2 : 0;
}

template <int WHAT, int K>
int sweep(uint64_t tstep, const char* out, long long cycles) {
  int dev = 0, sms = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, co_kernel<WHAT, K>, THREADS, 0);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, co_kernel<WHAT, K>);
  const int grid = sms * per_sm;
  launch<WHAT, K>(grid, 4, cycles / 10, tstep, nullptr, nullptr);  // warm-up
  FILE* f = fopen(out, "w");
  fprintf(f, "{\"what\": \"%s\", \"k\": %d, \"sqr\": %d, \"b3v\": %d, \"sms\": %d, \"blocks_per_sm\": %d, \"regs\": %d, "
             "\"cycles\": %lld, \"ue\": %d, \"ub\": %d, \"rows\": [",
          WHAT ? "encode" : "decode", K, WC_SQR, RT_B3V, sms, per_sm, fa.numRegs, cycles, UE, UB);
  for (int b3w = 0; b3w <= 8; ++b3w) {
    std::vector<Run> r;
    for (int rep = 0; rep < 3; ++rep) r.push_back(launch<WHAT, K>(grid, b3w, cycles, tstep, nullptr, nullptr));
    std::sort(r.begin(), r.end(), [](const Run& x, const Run& y) { return x.ms < y.ms; });
    const Run& m = r[1];
    fprintf(f, "%s{\"b3w\": %d, \"ms\": %.4f, \"compressions_per_s\": %.6e, \"words_per_s\": %.6e}", b3w ? ", " : "", b3w,
            m.ms, m.comp_per_s, m.words_per_s);
    printf("b3w %d: %.3f ms, %.4e compressions/s, %.4e words/s\n", b3w, m.ms, m.comp_per_s, m.words_per_s);
  }
  fprintf(f, "]}\n");
  fclose(f);
  return cudaGetLastError() ? 3 : 0;
}

}  // namespace rt

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: co gate|sweep WHAT K KEYFILE OUT [CYCLES]\n");
    return 64;
  }
  const std::string mode = argv[1];
  const int what = atoi(argv[2]), k = atoi(argv[3]);
  std::ifstream kf(argv[4]);
  int kk = 0;
  kf >> kk;
  if (kk != k || k < 1 || k > 3) return 65;
  uint64_t c[wc::MAXK] = {}, tau = 0, tstep = 0;
  for (int j = 0; j < k; ++j) kf >> c[j];
  kf >> tau >> tstep;
  if (!kf || wc_set_key(c, k, tau)) return 66;
  const uint32_t one = 1;
  if (cudaMemcpyToSymbol(rt::kOne, &one, sizeof one)) return 66;
  const long long cycles = argc > 6 ? atoll(argv[6]) : 400000000LL;
#define RT_CASE(W, K)                                                                   \
  if (what == W && k == K) return mode == "gate" ? rt::gate<W, K>(tstep, argv[5])       \
                                                 : rt::sweep<W, K>(tstep, argv[5], cycles);
  RT_CASE(1, 1) RT_CASE(1, 2) RT_CASE(1, 3) RT_CASE(0, 1) RT_CASE(0, 2) RT_CASE(0, 3)
#undef RT_CASE
  return 67;
}
