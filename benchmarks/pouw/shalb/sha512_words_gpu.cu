// The words committer on one GPU, minimal against served: N 16 KiB sha512/row/v2 chunks at positions POS0.., each
// hashed from P(131072)'s midstate and taken to its frame-v3 leaf of D_{f,W} (schema "sha512/row/v2/l1-w"), then the
// level-0 nodes over pairs of leaves. "minimal" is sha512_words.cuh (chain form, the first block from the prefix's
// constant midstate, the padding block as constants, the leaf's block 1 from round 15 and the node's block 2 from round
// 1 off the call's Pre). "served" is the committer's sha512.cuh (SHA_ROLL as built) in native_rows.cu's shapes: every
// block through sha::compress from the prefix's midstate, the padding block through compress_kw, the leaf's block 1
// through compress_from<15>, and the node from its first block's midstate. Rows are words as loaded (the message
// words), for both. Both kernels' leaves and nodes for the first NCHK chunks go to OUT with their rows, for
// shalb_words_check.py to hold to verity.ml.commitments bit for bit; each kernel is then timed (CUDA events, median of
// REPS launches).
//
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_90a -DSHA_ROLL=1 [-DWORDS_MINB=3] -I<committer> sha512_words_gpu.cu -o g
//   ./g DOMAIN_ID_FILE OUT
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sha512.cuh"
#include "sha512_words.cuh"

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e = (x);                                                              \
    if (e != cudaSuccess) {                                                           \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));       \
      exit(2);                                                                        \
    }                                                                                 \
  } while (0)

#ifndef WORDS_MINB  // the minimal kernels' blocks an SM (__launch_bounds__), which caps their registers a thread
#define WORDS_MINB 1
#endif

using u64 = uint64_t;
constexpr int N = 1 << 19, NCHK = 1024, REPS = 7, THREADS = 256;
constexpr u64 POS0 = 131072;

struct Served {
  u64 mid_p[8], kw_pad[80], m15[8], blk1[16], node_mid[8];
};

__global__ void fill(u64* rows, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    u64 z = i * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
    z = (z ^ z >> 30) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ z >> 27) * 0x94d049bb133111ebull;
    rows[i] = z ^ z >> 31;
  }
}

__global__ void __launch_bounds__(THREADS, WORDS_MINB)
    chunks_min(const __grid_constant__ shaw::Pre p, const u64* __restrict__ rows, u64* __restrict__ dig,
               u64* __restrict__ leaves, int n) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  u64 d[8], lf[8];
  shav2::chunk_loop<shaw::NB, shaw::PAD_BITS, shaw::Mid>(rows + (size_t)t * shaw::ROW_WORDS, d);
  shaw::leaf(p.leaf, POS0 + t, d, lf);
  for (int i = 0; i < 8; ++i) {
    dig[(size_t)t * 8 + i] = d[i];
    leaves[(size_t)t * 8 + i] = lf[i];
  }
}

__global__ void __launch_bounds__(THREADS, WORDS_MINB)
    nodes_min(const __grid_constant__ shaw::Pre p, const u64* __restrict__ leaves, u64* __restrict__ nodes, int m) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m) return;
  u64 nd[8];
  shaw::node(p, POS0 / 2 + j, leaves + (size_t)j * 16, leaves + (size_t)j * 16 + 8, nd);
  for (int i = 0; i < 8; ++i) nodes[(size_t)j * 8 + i] = nd[i];
}

__device__ __forceinline__ void leaf_words(u64 pos, const u64* d, u64* w) {
  w[0] = 3ull << 32 | pos << 8;
  w[1] = 0x1273;
  w[2] = 0x68613531322f726full;
  w[3] = 0x772f76322f6c312dull;
  w[4] = 0x7700000000000000ull;
  w[5] = 0x4000000000000000ull | d[0] >> 8;
#pragma unroll
  for (int i = 1; i < 8; ++i) w[5 + i] = d[i - 1] << 56 | d[i] >> 8;
  w[13] = d[7] << 56 | 0x80ull << 48;
  w[14] = 0;
  w[15] = 8 * 233;
}

__global__ void __launch_bounds__(THREADS) chunks_served(const __grid_constant__ Served k, const u64* __restrict__ rows,
                                                         u64* __restrict__ dig, u64* __restrict__ leaves, int n) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  u64 st[8], w[32];
  for (int i = 0; i < 8; ++i) st[i] = k.mid_p[i];
  const u64* r = rows + (size_t)t * shaw::ROW_WORDS;
#pragma unroll 1
  for (int b = 0; b < shaw::NB; ++b) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = r[16 * b + i];
    sha::compress(st, w);
  }
  sha::compress_kw(st, k.kw_pad);
  for (int i = 0; i < 8; ++i) dig[(size_t)t * 8 + i] = st[i];
#pragma unroll
  for (int i = 0; i < 15; ++i) w[i] = k.blk1[i];
  w[15] = 3ull << 56 | (POS0 + t) << 32;
  leaf_words(POS0 + t, st, w + 16);
  u64 out[8];
  sha::iv(out);
  sha::compress_from<15>(out, k.m15, w);
  sha::compress(out, w + 16);
  for (int i = 0; i < 8; ++i) leaves[(size_t)t * 8 + i] = out[i];
}

__global__ void __launch_bounds__(THREADS) nodes_served(const __grid_constant__ Served k, const u64* __restrict__ leaves,
                                                        u64* __restrict__ nodes, int m) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m) return;
  const u64 *l = leaves + (size_t)j * 16, *r = l + 8, idx = POS0 / 2 + j;
  u64 st[8], w[16];
  for (int i = 0; i < 8; ++i) st[i] = k.node_mid[i];
  w[0] = 3ull << 48 | idx << 24;
  w[1] = 0x40ull << 24 | l[0] >> 40;
#pragma unroll
  for (int i = 2; i < 9; ++i) w[i] = l[i - 2] << 24 | l[i - 1] >> 40;
  w[9] = l[7] << 24;
  w[10] = 0x40ull << 24 | r[0] >> 40;
#pragma unroll
  for (int i = 11; i < 16; ++i) w[i] = r[i - 11] << 24 | r[i - 10] >> 40;
  sha::compress(st, w);
  w[0] = r[5] << 24 | r[6] >> 40;
  w[1] = r[6] << 24 | r[7] >> 40;
  w[2] = r[7] << 24 | 0x80ull << 16;
#pragma unroll
  for (int i = 3; i < 15; ++i) w[i] = 0;
  w[15] = 8 * 277;
  sha::compress(st, w);
  for (int i = 0; i < 8; ++i) nodes[(size_t)j * 8 + i] = st[i];
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s DOMAIN_ID_FILE OUT\n", argv[0]);
    return 2;
  }
  uint8_t did[64];
  FILE* f = fopen(argv[1], "rb");
  if (!f || fread(did, 1, 64, f) != 64) return 2;
  fclose(f);
  shaw::Pre pre;
  shaw::pre_for(did, pre);
  Served sv;
  {
    constexpr shamin::RowV2Mid<shaw::ROW_BITS> mid{};
    for (int i = 0; i < 8; ++i) sv.mid_p[i] = mid.h[i];
    u64 pad[16] = {1ull << 63};
    pad[15] = shaw::PAD_BITS;
    sha::schedule_kw(pad, sv.kw_pad);
    u64 call[9], b1[16] = {};
    shal1::call_words(did, call);
    for (int i = 0; i < 6; ++i) b1[i] = shal1::B1(i);
    for (int i = 0; i < 9; ++i) b1[6 + i] = call[i];
    for (int i = 0; i < 16; ++i) sv.blk1[i] = b1[i];
    u64 iv[8];
    for (int i = 0; i < 8; ++i) iv[i] = shamin::IVt(i);
    sha::rounds_prefix(iv, b1, 15, sv.m15);
    for (int i = 0; i < 8; ++i) sv.node_mid[i] = pre.node_h[i];
    sv.node_mid[7] -= shamin::Kt(0);
  }
  int dev = 0, sms = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const size_t words = (size_t)N * shaw::ROW_WORDS;
  u64 *rows, *dig, *leaves, *nodes;
  CK(cudaMalloc(&rows, words * 8));
  CK(cudaMalloc(&dig, (size_t)N * 64));
  CK(cudaMalloc(&leaves, (size_t)N * 64));
  CK(cudaMalloc(&nodes, (size_t)N / 2 * 64));
  fill<<<sms * 8, 256>>>(rows, words);
  CK(cudaGetLastError());
  const int g = (N + THREADS - 1) / THREADS, gn = (N / 2 + THREADS - 1) / THREADS;
  std::vector<u64> h_rows((size_t)NCHK * shaw::ROW_WORDS), h((size_t)NCHK * 8);
  CK(cudaMemcpy(h_rows.data(), rows, h_rows.size() * 8, cudaMemcpyDeviceToHost));
  FILE* o = fopen(argv[2], "wb");
  const uint32_t hdr[4] = {NCHK, (uint32_t)POS0, (uint32_t)shaw::ROW_BYTES, 0};
  fwrite(hdr, 4, 4, o);
  fwrite(h_rows.data(), 8, h_rows.size(), o);
  for (int kind = 0; kind < 2; ++kind) {
    if (kind == 0) {
      chunks_min<<<g, THREADS>>>(pre, rows, dig, leaves, N);
      nodes_min<<<gn, THREADS>>>(pre, leaves, nodes, N / 2);
    } else {
      chunks_served<<<g, THREADS>>>(sv, rows, dig, leaves, N);
      nodes_served<<<gn, THREADS>>>(sv, leaves, nodes, N / 2);
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    for (u64* src : {dig, leaves}) {
      CK(cudaMemcpy(h.data(), src, h.size() * 8, cudaMemcpyDeviceToHost));
      fwrite(h.data(), 8, h.size(), o);
    }
    CK(cudaMemcpy(h.data(), nodes, h.size() / 2 * 8, cudaMemcpyDeviceToHost));
    fwrite(h.data(), 8, h.size() / 2, o);
  }
  fclose(o);
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  const char* names[4] = {"chunks+leaves minimal", "chunks+leaves served", "nodes minimal", "nodes served"};
  const double comps[4] = {(double)N * (shaw::NB + 3), (double)N * (shaw::NB + 3), (double)N, (double)N};
  for (int kind = 0; kind < 4; ++kind) {
    std::vector<float> ms;
    for (int r = 0; r < REPS + 1; ++r) {
      CK(cudaEventRecord(a));
      if (kind == 0) chunks_min<<<g, THREADS>>>(pre, rows, dig, leaves, N);
      else if (kind == 1) chunks_served<<<g, THREADS>>>(sv, rows, dig, leaves, N);
      else if (kind == 2) nodes_min<<<gn, THREADS>>>(pre, leaves, nodes, N / 2);
      else nodes_served<<<gn, THREADS>>>(sv, leaves, nodes, N / 2);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float m;
      CK(cudaEventElapsedTime(&m, a, b));
      if (r) ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    printf("time %s: median %.3f ms over %d launches (min %.3f, max %.3f), %.0f compressions, %.4f ns a compression, "
           "%d SMs\n", names[kind], med, REPS, ms.front(), ms.back(), comps[kind], med * 1e6 / comps[kind], sms);
  }
  return 0;
}
