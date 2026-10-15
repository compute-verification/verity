// The committed prefix word a block, on one sm_120 GPU: the honest int8 tensor-core step against the CUDA-core
// routes of the low-rank attack (attacks.py). Every kernel keeps its words in registers and updates each one once a
// block (L = 32 steps of k): the honest kernel with one m16n8k32 IMMA a 128 words, the attacks with
//   dp4a : T += dp4a(g, z)                       (r <= 4, G_s in one int8 slice; ceil(r/4) IDP.4A a word)
//   imad : T += sum_a g_a z_a                    (r IMADs a word, exact mod 2^32)
//   ffma : T += bits(magic + sum_a g_a z_a) - magic_bits   (r FFMAs and one IADD3 a word)
//   mix  : the honest kernel's 8 tiles and dp4a's 16 words in one warp (do the tensor and integer pipes overlap?)
// where g is the row's G_s = U D_s (changed every block) and z the word's column of Z (fixed). The integer kernels
// change g by an xor with the block index, so the compiler can't close the loop's sum into a formula. One CTA a SM
// (60 KB of dynamic shared memory pins it), clock64 around the block loop; each attack's words are checked bit for bit
// against the host's replay of the same arithmetic.
//
//   prefix <kind> <r> <warps> <blocks>     kind: honest | dp4a | imad | ffma | mix
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

constexpr int W = 16;               // words a lane (attacks)
constexpr int NT = 8;               // accumulator tiles a warp (honest): 4 words a lane each
constexpr float MAGIC = 12582912.0f;      // 1.5 * 2^23: bits(MAGIC + x) = 0x4B400000 + x for |x| < 2^22
constexpr int MAGIC_BITS = 0x4B400000;

__device__ __forceinline__ uint32_t lane_seed(int tid) { return 0x9E3779B9u * (uint32_t)(tid + 1); }

__global__ void k_honest(int blocks, int* out, long long* cyc) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t s = lane_seed(tid);
  uint32_t a0 = s, a1 = s * 3u, a2 = s * 5u, a3 = s * 7u, b0 = s * 11u, b1 = s * 13u;
  int acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; t++) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0;
  __syncthreads();
  long long t0 = clock64();
  for (int j = 0; j < blocks; j++) {
#pragma unroll
    for (int t = 0; t < NT; t++)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                   "{%0,%1,%2,%3};"
                   : "+r"(acc[t][0]), "+r"(acc[t][1]), "+r"(acc[t][2]), "+r"(acc[t][3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    a0 += 0x01010101u;              // the next block's operands (one op a lane a block, over 4*NT words)
  }
  __syncthreads();
  long long t1 = clock64();
  int x = 0;
#pragma unroll
  for (int t = 0; t < NT; t++) x ^= acc[t][0] ^ acc[t][1] ^ acc[t][2] ^ acc[t][3];
  out[tid] = x;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

// Host and device share these: a lane's g_0 and z values.
__host__ __device__ inline uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu;
  return x ^ (x >> 16); }
__host__ __device__ inline int pm1(uint32_t x) { return (x & 1u) ? 1 : -1; }
__host__ __device__ inline int g0_of(int tid, int a) { return (int)(mix(tid * 64 + a) % 61u) - 30; }
__host__ __device__ inline int z_of(int tid, int w, int a) { return pm1(mix(0x51ED270Bu + tid * 1024 + w * 16 + a)); }
__host__ __device__ inline uint32_t pack4(int v0, int v1, int v2, int v3) {
  return (uint32_t)(v0 & 255) | ((uint32_t)(v1 & 255) << 8) | ((uint32_t)(v2 & 255) << 16) | ((uint32_t)(v3 & 255) << 24);
}

template <int R4>
__global__ void k_dp4a(int r, int blocks, int* out, long long* cyc) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t g[R4], z[W][R4];
#pragma unroll
  for (int q = 0; q < R4; q++) {
    int v[4];
    for (int b = 0; b < 4; b++) v[b] = (4 * q + b < r) ? g0_of(tid, 4 * q + b) : 0;
    g[q] = pack4(v[0], v[1], v[2], v[3]);
#pragma unroll
    for (int w = 0; w < W; w++) {
      int u[4];
      for (int b = 0; b < 4; b++) u[b] = (4 * q + b < r) ? z_of(tid, w, 4 * q + b) : 0;
      z[w][q] = pack4(u[0], u[1], u[2], u[3]);
    }
  }
  int T[W];
#pragma unroll
  for (int w = 0; w < W; w++) T[w] = 0;
  __syncthreads();
  long long t0 = clock64();
  for (int j = 0; j < blocks; j++) {
#pragma unroll
    for (int w = 0; w < W; w++)
#pragma unroll
      for (int q = 0; q < R4; q++) T[w] = __dp4a((int)g[q], (int)z[w][q], T[w]);
#pragma unroll
    for (int q = 0; q < R4; q++) g[q] ^= (uint32_t)(j + 1);    // G_{s+1}: one op a lane a block, over W words
  }
  __syncthreads();
  long long t1 = clock64();
#pragma unroll
  for (int w = 0; w < W; w++) out[tid * W + w] = T[w];
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

template <int R>
__global__ void k_imad(int blocks, int* out, long long* cyc) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t g[R], z[W][R];
#pragma unroll
  for (int a = 0; a < R; a++) {
    g[a] = (uint32_t)g0_of(tid, a);
#pragma unroll
    for (int w = 0; w < W; w++) z[w][a] = (uint32_t)z_of(tid, w, a);
  }
  uint32_t T[W];
#pragma unroll
  for (int w = 0; w < W; w++) T[w] = 0;
  __syncthreads();
  long long t0 = clock64();
  for (int j = 0; j < blocks; j++) {
#pragma unroll
    for (int w = 0; w < W; w++)
#pragma unroll
      for (int a = 0; a < R; a++) T[w] += g[a] * z[w][a];
#pragma unroll
    for (int a = 0; a < R; a++) g[a] ^= (uint32_t)(j + 1);
  }
  __syncthreads();
  long long t1 = clock64();
#pragma unroll
  for (int w = 0; w < W; w++) out[tid * W + w] = (int)T[w];
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

// The honest kernel's NT tiles (the attack's tensor share, k padded to 32) beside k_dp4a<1>'s W words, one warp.
__global__ void k_mix(int r, int blocks, int* out, int* out_t, long long* cyc) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t s = lane_seed(tid);
  uint32_t a0 = s, a1 = s * 3u, a2 = s * 5u, a3 = s * 7u, b0 = s * 11u, b1 = s * 13u;
  int acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; t++) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0;
  uint32_t g, z[W];
  {
    int v[4];
    for (int b = 0; b < 4; b++) v[b] = (b < r) ? g0_of(tid, b) : 0;
    g = pack4(v[0], v[1], v[2], v[3]);
#pragma unroll
    for (int w = 0; w < W; w++) {
      int u[4];
      for (int b = 0; b < 4; b++) u[b] = (b < r) ? z_of(tid, w, b) : 0;
      z[w] = pack4(u[0], u[1], u[2], u[3]);
    }
  }
  int T[W];
#pragma unroll
  for (int w = 0; w < W; w++) T[w] = 0;
  __syncthreads();
  long long t0 = clock64();
  for (int j = 0; j < blocks; j++) {
#pragma unroll
    for (int t = 0; t < NT; t++)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                   "{%0,%1,%2,%3};"
                   : "+r"(acc[t][0]), "+r"(acc[t][1]), "+r"(acc[t][2]), "+r"(acc[t][3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    a0 += 0x01010101u;
#pragma unroll
    for (int w = 0; w < W; w++) T[w] = __dp4a((int)g, (int)z[w], T[w]);
    g ^= (uint32_t)(j + 1);
  }
  __syncthreads();
  long long t1 = clock64();
  int x = 0;
#pragma unroll
  for (int t = 0; t < NT; t++) x ^= acc[t][0] ^ acc[t][1] ^ acc[t][2] ^ acc[t][3];
  out_t[tid] = x;
#pragma unroll
  for (int w = 0; w < W; w++) out[tid * W + w] = T[w];
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

template <int R>
__global__ void k_ffma(int blocks, int* out, long long* cyc) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float g[R], z[W][R];
#pragma unroll
  for (int a = 0; a < R; a++) {
    g[a] = (float)g0_of(tid, a);
#pragma unroll
    for (int w = 0; w < W; w++) z[w][a] = (float)z_of(tid, w, a);
  }
  int T[W];
#pragma unroll
  for (int w = 0; w < W; w++) T[w] = 0;
  __syncthreads();
  long long t0 = clock64();
  for (int j = 0; j < blocks; j++) {
#pragma unroll
    for (int w = 0; w < W; w++) {
      float acc = __fmaf_rn(g[0], z[w][0], MAGIC);
#pragma unroll
      for (int a = 1; a < R; a++) acc = __fmaf_rn(g[a], z[w][a], acc);
      T[w] += __float_as_int(acc) - MAGIC_BITS;
    }
#pragma unroll
    for (int a = 0; a < R; a++) g[a] += 1.0f;
  }
  __syncthreads();
  long long t1 = clock64();
#pragma unroll
  for (int w = 0; w < W; w++) out[tid * W + w] = T[w];
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

static int check(const char* kind, int r, int lanes, int blocks, const std::vector<int>& got) {
  int bad = 0;
  for (int tid = 0; tid < lanes; tid++) {
    for (int w = 0; w < W; w++) {
      uint32_t T = 0;
      int R4 = (r + 3) / 4, zv[4 * 4], gv[4 * 4];
      for (int a = 0; a < 4 * R4; a++) {
        zv[a] = a < r ? z_of(tid, w, a) : 0;
        gv[a] = a < r ? g0_of(tid, a) : 0;
      }
      if (!strcmp(kind, "dp4a") || !strcmp(kind, "mix")) {
        std::vector<uint32_t> g(R4);
        for (int q = 0; q < R4; q++) g[q] = pack4(gv[4 * q], gv[4 * q + 1], gv[4 * q + 2], gv[4 * q + 3]);
        for (int j = 0; j < blocks; j++) {
          for (int q = 0; q < R4; q++)
            for (int b = 0; b < 4; b++) T += (uint32_t)((int8_t)((g[q] >> (8 * b)) & 255) * zv[4 * q + b]);
          for (int q = 0; q < R4; q++) g[q] ^= (uint32_t)(j + 1);
        }
      } else if (!strcmp(kind, "imad")) {
        uint32_t g[4 * 4];
        for (int a = 0; a < r; a++) g[a] = (uint32_t)gv[a];
        for (int j = 0; j < blocks; j++) {
          for (int a = 0; a < r; a++) T += g[a] * (uint32_t)zv[a];
          for (int a = 0; a < r; a++) g[a] ^= (uint32_t)(j + 1);
        }
      } else {
        for (int j = 0; j < blocks; j++)
          for (int a = 0; a < r; a++) T += (uint32_t)((gv[a] + j) * zv[a]);
      }
      if ((int)T != got[(size_t)tid * W + w]) bad++;
    }
  }
  return bad;
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: prefix honest|dp4a|imad|ffma|mix r warps blocks\n"); return 2; }
  const char* kind = argv[1];
  int r = atoi(argv[2]), warps = atoi(argv[3]), blocks = atoi(argv[4]);
  int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  int threads = warps * 32, lanes = sms * threads;
  size_t smem = 60 * 1024;
  int *d_out, *d_out_t; long long* d_cyc;
  CK(cudaMalloc(&d_out, (size_t)lanes * W * sizeof(int)));
  CK(cudaMalloc(&d_out_t, (size_t)lanes * sizeof(int)));
  CK(cudaMalloc(&d_cyc, sms * sizeof(long long)));
  auto launch = [&](int nb) {
    if (!strcmp(kind, "honest")) {
      CK(cudaFuncSetAttribute(k_honest, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      k_honest<<<sms, threads, smem>>>(nb, d_out, d_cyc);
    }
    else if (!strcmp(kind, "mix") && r <= 4) {
      CK(cudaFuncSetAttribute(k_mix, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      k_mix<<<sms, threads, smem>>>(r, nb, d_out, d_out_t, d_cyc);
    }
#define DP(R4) else if (!strcmp(kind, "dp4a") && (r + 3) / 4 == R4) { \
      CK(cudaFuncSetAttribute(k_dp4a<R4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); \
      k_dp4a<R4><<<sms, threads, smem>>>(r, nb, d_out, d_cyc); }
    DP(1) DP(2)
#define IM(R) else if (!strcmp(kind, "imad") && r == R) { \
      CK(cudaFuncSetAttribute(k_imad<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); \
      k_imad<R><<<sms, threads, smem>>>(nb, d_out, d_cyc); }
    IM(1) IM(2) IM(4)
#define FF(R) else if (!strcmp(kind, "ffma") && r == R) { \
      CK(cudaFuncSetAttribute(k_ffma<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); \
      k_ffma<R><<<sms, threads, smem>>>(nb, d_out, d_cyc); }
    FF(1) FF(2) FF(3) FF(4) FF(8)
    else { fprintf(stderr, "no kernel for %s r=%d\n", kind, r); exit(2); }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
  };
  int wpl = !strcmp(kind, "honest") ? 4 * NT : !strcmp(kind, "mix") ? 4 * NT + W : W;
  int bad = -1;
  if (strcmp(kind, "honest")) {
    int nb = 64;                    // check: a short run replayed on the host (FP32 stays below 2^22)
    launch(nb);
    std::vector<int> got((size_t)lanes * W);
    CK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
    bad = check(kind, r, lanes, nb, got);
  }
  launch(blocks);                   // warm
  double best = 1e30, sum = 0;
  const int reps = 5;
  for (int rep = 0; rep < reps; rep++) {
    launch(blocks);
    std::vector<long long> cyc(sms);
    CK(cudaMemcpy(cyc.data(), d_cyc, sms * sizeof(long long), cudaMemcpyDeviceToHost));
    long long mx = 0;
    for (long long c : cyc) mx = c > mx ? c : mx;
    double words = (double)threads * wpl * blocks;   // a CTA (one a SM)
    double cpw = (double)mx / words;   // SM clocks a committed word a block
    best = cpw < best ? cpw : best;
    sum += cpw;
  }
  // units: int8 tensor MAC slots, 1,024 a SM clock
  printf("{\"kind\":\"%s\",\"r\":%d,\"warps\":%d,\"blocks\":%d,\"sms\":%d,\"words_per_lane\":%d,"
         "\"sm_clk_per_word_block\":%.6g,\"units_per_word_block\":%.4g,\"units_mean\":%.4g,\"mismatches\":%d}\n",
         kind, r, warps, blocks, sms, wpl, best, best * 1024, sum / reps * 1024, bad);
  return bad > 0 ? 1 : 0;
}
