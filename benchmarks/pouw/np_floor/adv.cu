// The cheapest program for ssn_1l's committed words given the data's partial dots for free (MH100w makes every
// salt-independent function of the committed input free): word(i,j,t) = D'[i,t,j] + G[p_i, j] + P[i,t], where
// D' = D_t − β_j (precomputed), G[p, j] = Σ_e p_e·C[e, j] (public, 16 patterns), p_i = row i's four step-1 coin bits and
// P[i,t] = Σ_{2≤t'≤t} a_t'[i] (the coin prefix). In MH100w that is one IADD3 a word (the loads are free moves): 32
// a word, the honest wgmma's price. This times it on the H200 with D' in HBM, in L2 and in shared memory, folding each
// word into a sink (one LOP3, the stand-in for the hash's read), and checks every word of the first rows on the host.
//   nvcc -O3 -arch=sm_90a -std=c++17 adv.cu -o adv
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

constexpr double UNITS_PER_S = 540672.0 * 1.98e9;

// One thread a column j (coalesced), every row i and step t of its block's rows. dump != null: write the words.
__global__ void adv_words(const int* __restrict__ D, const int* __restrict__ G, const int* __restrict__ P,
                          const unsigned char* __restrict__ pat, int m, int S, int n, unsigned* sink, int* dump) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  for (int i = blockIdx.y; i < m; i += gridDim.y) {
    const int g = G[pat[i] * n + j];
    const int* d = D + (size_t)i * S * n + j;
    const int* p = P + (size_t)i * S;
#pragma unroll 8
    for (int t = 0; t < S; t++) {
      const int w = d[(size_t)t * n] + g + p[t];
      acc ^= (unsigned)w;
      if (dump) dump[((size_t)i * S + t) * n + j] = w;
    }
  }
  sink[(blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x] = acc;
}

// D' resident in shared memory: 2 rows x S steps x 256 columns of a block's tile, re-read REPS times (the bound when
// the free values sit next to the SM).
__global__ void adv_words_smem(const int* __restrict__ D, const int* __restrict__ G, const int* __restrict__ P,
                               int S, int n, int reps, unsigned* sink) {
  extern __shared__ int sd[];
  const int j0 = blockIdx.x * 256, tid = threadIdx.x;
  for (int k = tid; k < 2 * S * 256; k += blockDim.x) {
    const int r = k / (S * 256), t = (k / 256) % S, c = k % 256;
    sd[k] = D[((size_t)r * S + t) * n + j0 + c];
  }
  __syncthreads();
  const int c = tid & 255, r = tid >> 8;  // 512 threads: 2 rows x 256 columns
  const int g = G[(r & 15) * n + j0 + c];
  unsigned acc = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; rep++) {
#pragma unroll 8
    for (int t = 0; t < S; t++) acc ^= (unsigned)(sd[(r * S + t) * 256 + c] + g + P[r * S + t] + rep);
  }
  sink[blockIdx.x * blockDim.x + tid] = acc;
}

int main() {
  // Llama-3.2-1B's qkv-like shape at n = 2,048, K = 2,048: S = 1 + ceil((2048 - 28)/31) = 67 steps
  const int n = 2048, S = 67;
  std::vector<int> C(4 * n);  // the public code: C[e, j] = ((c_j // 13^e) mod 13) − 6, c_j = j + 1
  for (int j = 0; j < n; j++) {
    int c = j + 1;
    for (int e = 0; e < 4; e++) C[e * n + j] = c % 13 - 6, c /= 13;
  }
  std::vector<int> G(16 * n);
  for (int p = 0; p < 16; p++)
    for (int j = 0; j < n; j++) {
      int s = 0;
      for (int e = 0; e < 4; e++) s += ((p >> e) & 1) * C[e * n + j];
      G[p * n + j] = s;
    }
  printf("{");
  bool first = true;
  for (int m : {256, 2048}) {  // 2048 rows: D' is 1.1 GB (HBM); 256 rows: 140 MB (HBM, L2-spilling)
    size_t nw = (size_t)m * S * n;
    std::vector<int> D(nw), P((size_t)m * S);
    std::vector<unsigned char> pat(m);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { return (x ^= x << 13, x ^= x >> 7, x ^= x << 17, x); };
    for (size_t k = 0; k < nw; k++) D[k] = (int)(rnd() % 1000003) - 500001;
    for (int i = 0; i < m; i++) {
      pat[i] = rnd() & 15;
      int s = 0;
      for (int t = 0; t < S; t++) s += t ? (int)(rnd() & 1) : 0, P[(size_t)i * S + t] = s;
    }
    int *dD, *dG, *dP, *dump;
    unsigned char* dpat;
    unsigned* sink;
    CK(cudaMalloc(&dD, nw * 4));
    CK(cudaMalloc(&dG, G.size() * 4));
    CK(cudaMalloc(&dP, P.size() * 4));
    CK(cudaMalloc(&dpat, m));
    CK(cudaMalloc(&sink, 64u << 20));
    CK(cudaMemcpy(dD, D.data(), nw * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dG, G.data(), G.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dP, P.data(), P.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpat, pat.data(), m, cudaMemcpyHostToDevice));
    dim3 grid(n / 256, 264);
    size_t bad = 0;
    if (m == 256) {  // check every word against the host formula
      CK(cudaMalloc(&dump, nw * 4));
      adv_words<<<grid, 256>>>(dD, dG, dP, dpat, m, S, n, sink, dump);
      CK(cudaDeviceSynchronize());
      std::vector<int> h(nw);
      CK(cudaMemcpy(h.data(), dump, nw * 4, cudaMemcpyDeviceToHost));
      for (int i = 0; i < m; i++)
        for (int t = 0; t < S; t++)
          for (int j = 0; j < n; j++) {
            size_t k = ((size_t)i * S + t) * n + j;
            bad += h[k] != D[k] + G[pat[i] * n + j] + P[(size_t)i * S + t];
          }
      CK(cudaFree(dump));
    }
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    adv_words<<<grid, 256>>>(dD, dG, dP, dpat, m, S, n, sink, nullptr);
    CK(cudaEventRecord(a));
    const int R = 5;
    for (int r = 0; r < R; r++) adv_words<<<grid, 256>>>(dD, dG, dP, dpat, m, S, n, sink, nullptr);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    double wps = (double)nw * R / (ms * 1e-3);
    printf("%s\"hbm_m%d\": {\"words\": %zu, \"words_per_s\": %.6g, \"units_per_word\": %.2f, \"bad\": %zu}",
           first ? "" : ", ", m, nw, wps, UNITS_PER_S / wps, bad);
    first = false;
    if (m == 256) {
      const int smem = 2 * S * 256 * 4, reps = 64;
      CK(cudaFuncSetAttribute(adv_words_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      adv_words_smem<<<n / 256 * 33, 512, smem>>>(dD, dG, dP, S, n, 1, sink);
      CK(cudaEventRecord(a));
      adv_words_smem<<<n / 256 * 33, 512, smem>>>(dD, dG, dP, S, n, reps, sink);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      CK(cudaEventElapsedTime(&ms, a, b));
      double w = (double)(n / 256 * 33) * 512 * S * reps;
      printf(", \"smem\": {\"words\": %.0f, \"words_per_s\": %.6g, \"units_per_word\": %.2f}", w, w / (ms * 1e-3),
             UNITS_PER_S / (w / (ms * 1e-3)));
    }
    CK(cudaFree(dD));
    CK(cudaFree(dG));
    CK(cudaFree(dP));
    CK(cudaFree(dpat));
    CK(cudaFree(sink));
  }
  printf("}\n");
  return 0;
}
