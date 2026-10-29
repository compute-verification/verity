// A2u beside a saturating issue stream: does a uniform-datapath instruction take an SMSP issue slot when the lanes
// alone issue at nearly one warp instruction a cycle? opcode_probe's a2u.cu put its uniform chains beside streams
// that issue at 0.79-0.88 alone, so a fifth of the slots were idle. Here the stream is
// 64 FFMA a loop over NF independent accumulators (the FMA pipe takes one a cycle; nothing else binds), one block an SM
// of 1024, 256 or 128 threads (8, 2 or 1 warps a SMSP; at most 64 registers a thread), beside NP pairs of integer chains
// (UIMAD + ULOP3 where ptxas puts a pair on the uniform datapath), K chain instructions a loop:
//   s_<NF>_0          the stream alone;
//   u_<NF>_<K>_<NP>   chains seeded from a kernel parameter (uniform: ptxas may place them on the uniform datapath);
//   l_<NF>_<K>_<NP>   the same chains seeded from a per-lane load (on lanes: the control);
//   o<k>_16_64_2      uniform chains of other opcodes (k = 1: ULEA, UIADD3; 2: USHF, UPRMT; 3: UIMNMX, UISETP, USEL);
//   t_16_<NI>         NI IMMA.16832 a loop beside the stream (the tensor pipe takes 16 SMSP-cycles each, below the
//                     stream's issue, so an IMMA dual issued beside the lanes would show as a rate above 1);
//   tu_16_<NI>        the IMMAs and 64 UIMAD/ULOP3 chain instructions beside the stream.
// a2u_sat_count.py counts each loop body's uniform (U*) and vector instructions; with A2u (every instruction an issue
// slot) a loop takes at least its instruction count in SMSP-cycles, so a total rate above 1 a SMSP-cycle refutes it,
// and so does an l_ kernel's (vector beside vector) rate above 1.
#include <stdint.h>
#include <stdio.h>

#define NT 1024

template <int NF, int K, int NP, bool UNI, int OPK = 0, int NI = 0>
__device__ __forceinline__ void a2u(uint32_t s, const uint32_t *lane_seed, float g, float *out, long long *cyc,
                                    int iters) {
  float f[NF];
#pragma unroll
  for (int c = 0; c < NF; ++c) f[c] = (float)(threadIdx.x + c);
  uint32_t u[2 * NP];
  const uint32_t s0 = UNI ? s : lane_seed[blockIdx.x * blockDim.x + threadIdx.x];
#pragma unroll
  for (int c = 0; c < 2 * NP; ++c) u[c] = s0 * (2u * c + 3u) + (uint32_t)c;
  uint32_t ma[4], mb[2];
  int mc[NI > 0 ? NI : 1][4];
  if (NI > 0) {
    const uint32_t m0 = lane_seed[blockIdx.x * blockDim.x + threadIdx.x] ^ threadIdx.x;
#pragma unroll
    for (int c = 0; c < 4; ++c) ma[c] = m0 * (2u * c + 5u);
#pragma unroll
    for (int c = 0; c < 2; ++c) mb[c] = m0 * (2u * c + 13u);
#pragma unroll
    for (int j = 0; j < (NI > 0 ? NI : 1); ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) mc[j][c] = j + c;
  }
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int q = 0; q < 64; ++q) f[q % NF] = __fmaf_rn(f[q % NF], g, 1.0f);
#pragma unroll
    for (int j = 0; j < NI; ++j)
      asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+r"(mc[j][0]), "+r"(mc[j][1]), "+r"(mc[j][2]), "+r"(mc[j][3])
          : "r"(ma[0]), "r"(ma[1]), "r"(ma[2]), "r"(ma[3]), "r"(mb[0]), "r"(mb[1]));
#pragma unroll
    for (int i = 0; i < K / (2 * NP); ++i)
#pragma unroll
      for (int p = 0; p < 2 * NP; p += 2) {
        const uint32_t k = 0x7F4A7C15u + (uint32_t)(i * 8 + p) * 0x9E3779B9u;
        if constexpr (OPK == 0) {  // UIMAD, ULOP3
          u[p] = u[p] * 0x9E3779B1u + u[p + 1];
          u[p + 1] = u[p + 1] ^ u[p] ^ k;
        } else if constexpr (OPK == 1) {  // ULEA, UIADD3
          u[p] = (u[p] << 5) + u[p + 1];
          u[p + 1] = u[p + 1] + u[p] + k;
        } else if constexpr (OPK == 2) {  // USHF funnel, UPRMT
          u[p] = __funnelshift_l(u[p + 1], u[p], 7);
          u[p + 1] = __byte_perm(u[p + 1], u[p], 0x6427);
        } else {  // UIMNMX, UISETP, USEL
          u[p] = max(u[p], u[p + 1]) ^ k;
          u[p + 1] = u[p] > u[p + 1] ? u[p] : u[p + 1] + k;
        }
      }
  }
  __syncthreads();
  long long t1 = clock64();
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < NF; ++c) acc += f[c];
  uint32_t x = 0;
#pragma unroll
  for (int c = 0; c < 2 * NP; ++c) x ^= u[c];
#pragma unroll
  for (int j = 0; j < NI; ++j)
#pragma unroll
    for (int c = 0; c < 4; ++c) x ^= (uint32_t)mc[j][c];
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc + (float)(x & 1u);
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

#define A2U(NF, K, NP)                                                                                       \
  extern "C" __global__ void __launch_bounds__(NT, 1) u_##NF##_##K##_##NP(uint32_t s, const uint32_t *ls, float g, \
                                                                         float *o, long long *c, int n) {    \
    a2u<NF, K, NP, true>(s, ls, g, o, c, n);                                                                 \
  }                                                                                                          \
  extern "C" __global__ void __launch_bounds__(NT, 1) l_##NF##_##K##_##NP(uint32_t s, const uint32_t *ls, float g, \
                                                                         float *o, long long *c, int n) {    \
    a2u<NF, K, NP, false>(s, ls, g, o, c, n);                                                                \
  }
#define BASE(NF)                                                                                            \
  extern "C" __global__ void __launch_bounds__(NT, 1) s_##NF##_0(uint32_t s, const uint32_t *ls, float g, float *o, \
                                                                long long *c, int n) {                      \
    a2u<NF, 0, 1, true>(s, ls, g, o, c, n);                                                                 \
  }

#define OPS(OPK, NF, K, NP)                                                                                 \
  extern "C" __global__ void __launch_bounds__(NT, 1) o##OPK##_##NF##_##K##_##NP(                          \
      uint32_t s, const uint32_t *ls, float g, float *o, long long *c, int n) {                             \
    a2u<NF, K, NP, true, OPK>(s, ls, g, o, c, n);                                                           \
  }
#define MMA(NF, NI)                                                                                         \
  extern "C" __global__ void __launch_bounds__(NT, 1) t_##NF##_##NI(uint32_t s, const uint32_t *ls, float g, \
                                                                   float *o, long long *c, int n) {         \
    a2u<NF, 0, 1, true, 0, NI>(s, ls, g, o, c, n);                                                          \
  }                                                                                                         \
  extern "C" __global__ void __launch_bounds__(NT, 1) tu_##NF##_##NI(uint32_t s, const uint32_t *ls, float g, \
                                                                    float *o, long long *c, int n) {        \
    a2u<NF, 64, 2, true, 0, NI>(s, ls, g, o, c, n);                                                         \
  }

BASE(16)
BASE(40)
OPS(1, 16, 64, 2)
OPS(2, 16, 64, 2)
OPS(3, 16, 64, 2)
MMA(16, 2)
MMA(16, 3)
MMA(16, 4)
A2U(16, 32, 2)
A2U(16, 64, 2)
A2U(16, 64, 4)
A2U(16, 128, 4)
A2U(40, 64, 4)
A2U(40, 128, 4)
A2U(40, 96, 3)
A2U(40, 192, 3)
A2U(44, 128, 4)
A2U(44, 256, 4)

#ifndef __CUDA_ARCH__
#include <cstdlib>
#include <cuda_runtime.h>

typedef void (*Kern)(uint32_t, const uint32_t *, float, float *, long long *, int);

int main() {
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  printf("device cc=%d.%d sms=%d\n", major, minor, sms);
  struct { const char *name; Kern k; } ks[] = {
#define E(n) {#n, n}
      E(s_16_0),        E(s_40_0),        E(u_16_32_2),     E(l_16_32_2),     E(u_16_64_2),    E(l_16_64_2),
      E(u_16_64_4),     E(l_16_64_4),     E(u_16_128_4),    E(l_16_128_4),    E(u_40_64_4),    E(l_40_64_4),
      E(u_40_128_4),    E(l_40_128_4),    E(u_40_96_3),     E(l_40_96_3),     E(u_40_192_3),   E(l_40_192_3),
      E(u_44_128_4),    E(l_44_128_4),    E(u_44_256_4),    E(l_44_256_4),    E(o1_16_64_2),   E(o2_16_64_2),
      E(o3_16_64_2),    E(t_16_2),        E(t_16_3),        E(t_16_4),        E(tu_16_2),      E(tu_16_3),
      E(tu_16_4),
  };
  const int ITERS = 4096, blocks = sms;
  uint32_t *d_ls;
  float *d_out;
  long long *d_cyc;
  cudaMalloc(&d_ls, (size_t)blocks * NT * sizeof(uint32_t));
  cudaMemset(d_ls, 0x5e, (size_t)blocks * NT * sizeof(uint32_t));
  cudaMalloc(&d_out, (size_t)blocks * NT * sizeof(float));
  cudaMalloc(&d_cyc, blocks * sizeof(long long));
  long long *cyc = (long long *)malloc(blocks * sizeof(long long));
  int bad = 0;
  // one block an SM of 128 w threads: w warps a SMSP; SMSP-cycles a warp-loop is the block's cycles over w warps' loops
  const int warps[] = {8, 2, 1};
  for (int w : warps)
    for (auto &e : ks) {
      for (int rep = 0; rep < 3; ++rep) {
        e.k<<<blocks, 128 * w>>>(0x5e5e5e5eu, d_ls, 1.0000001f, d_out, d_cyc, ITERS);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        bad |= err != cudaSuccess;
        cudaMemcpy(cyc, d_cyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
        long long mx = 0;
        for (int b = 0; b < blocks; ++b) mx = cyc[b] > mx ? cyc[b] : mx;
        printf("a2u %s w=%d rep=%d smsp_cyc_per_warp_loop=%.4f err=%s\n", e.name, w, rep,
               (double)mx / ((double)w * ITERS), cudaGetErrorString(err));
      }
    }
  return bad ? 3 : 0;
}
#endif
