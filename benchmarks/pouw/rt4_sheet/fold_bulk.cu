// RT4-SHEET-1: fx1's fold on sm_120 with its XOR on the bulk engine, K logical warps a thread, against FOLD-OPT's
// N_w φ* = 16 at π* = DS_issue. This is cpw_routes.cu's mode 10 (`cpwx`, RT3-120: the block's y slot in global memory,
// each lane's p fk staged in shared memory, thread 0 bulk-reducing (cp.reduce.async.bulk .xor) that slot and then the
// static KK slot into the global one, every lane reading u = y ^ p fk ^ KK back past L1), with two changes a cheater
// can make:
//   - each thread carries K logical warps' state (y, z and the 8 IMMA.16832 accumulators of each), so one pair of
//     block barriers, one fence and one thread-0 body serve 2 K units a trip instead of 2;
//   - WIDE = 1 moves y and u in one 256-bit st.global.v8 / ld.global.cg.v8 (STG.E.ENL2.256, LDG.E.ENL2.256 on sm_120a,
//     width_probe.sh) and p fk in two STS.128, where mode 10 takes two STG.128, four STS.64 and two LDG.128.
// K BT = 512 for every variant, so the bulk size (16 KB a reduce) and the shared memory (80 KB) are mode 10's.
// A unit is one IMMA.16832 and fx1's absorb of its 128 words (`segment._fx1_absorb`); a set is 8 units a logical warp.
// `main` runs on the device it finds: bitexact (one block, every logical lane's final state against the host's
// reference after 64 sets), then tp at 1 block a SM and at full occupancy: SMSP-cycles a unit (clock64 a block, max
// over blocks).
#include <stdint.h>
#include <stdio.h>

#define NCH 8
#define CPS 64

__host__ __device__ constexpr uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
__host__ __device__ constexpr uint32_t KA(int t) { return mix32(0xA0000000u + 2u * t) | 1u; }
__host__ __device__ constexpr uint32_t KK(int t) { return mix32(0xB0000000u + 2u * t + 1u); }
__host__ __device__ constexpr int ROT(int t) { return ((t * 7 + 5) % 31) + 1; }
__host__ __device__ constexpr uint32_t FK(int q) { return mix32(0xA0000000u + 2u * (16u + 32u * q)) | 1u; }

__device__ __forceinline__ void imma(uint32_t (&d)[4], uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a), "r"(b));
}

__device__ __forceinline__ void st128(uint32_t *p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void st256(uint32_t *p, const uint32_t (&v)[NCH]) {
  asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(p), "r"(v[0]), "r"(v[1]), "r"(v[2]),
               "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7])
               : "memory");
}
__device__ __forceinline__ void sts64(uint32_t addr, uint32_t a, uint32_t b) {
  asm volatile("st.shared.v2.b32 [%0], {%1,%2};" ::"r"(addr), "r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ void sts128(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void ldcg128(const uint32_t *p, uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
  asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
}
__device__ __forceinline__ void ldcg256(const uint32_t *p, uint32_t (&v)[NCH]) {
  asm volatile("ld.global.cg.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
               : "l"(p)
               : "memory");
}
__device__ __forceinline__ void bulk_xor_n(uint32_t *g, uint32_t s, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%1], %2;\n\t"
               "cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group 0;" ::"l"(g), "r"(s), "r"(bytes)
               : "memory");
}

// one round trip, t = 2H and 2H + 1 of every chain of every logical warp: logical warp k of thread tid owns word slot
// k BT + tid of the block's slot, NCH words, in global memory (y, then u) and in shared memory (p fk)
template <int K, int BT, int WIDE, int H>
__device__ __forceinline__ void trip(uint32_t (&y)[K][NCH], uint32_t (&z)[K][NCH], const uint32_t (&d)[K][8][4],
                                     uint32_t *gb, uint32_t sd, uint32_t sk, uint32_t bytes, int tid) {
#pragma unroll
  for (int k = 0; k < K; ++k) {
    uint32_t m[NCH];
#pragma unroll
    for (int j = 0; j < NCH; ++j) m[j] = d[k][2 * H + j / 4][j % 4] * FK(4 * (2 * H + j / 4) + j % 4);
    uint32_t *g = gb + NCH * (k * BT + tid);
    const uint32_t s = sd + 4u * NCH * (k * BT + tid);
    if (WIDE) {
      st256(g, y[k]);
      sts128(s, m[0], m[1], m[2], m[3]);
      sts128(s + 16, m[4], m[5], m[6], m[7]);
    } else {
      st128(g, y[k][0], y[k][1], y[k][2], y[k][3]);
      st128(g + 4, y[k][4], y[k][5], y[k][6], y[k][7]);
      sts64(s, m[0], m[1]);
      sts64(s + 8, m[2], m[3]);
      sts64(s + 16, m[4], m[5]);
      sts64(s + 24, m[6], m[7]);
    }
  }
  asm volatile("fence.proxy.async;" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    bulk_xor_n(gb, sd, bytes);
    bulk_xor_n(gb, sk, bytes);
    asm volatile("fence.proxy.async;" ::: "memory");
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const uint32_t *g = gb + NCH * (k * BT + tid);
    uint32_t u[NCH];
    if (WIDE) {
      ldcg256(g, u);
    } else {
      ldcg128(g, u[0], u[1], u[2], u[3]);
      ldcg128(g + 4, u[4], u[5], u[6], u[7]);
    }
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      y[k][j] = u[j] * KA(2 * H + j / 4) + z[k][j];
      z[k][j] = __funnelshift_l(u[j], u[j], ROT(2 * H + j / 4)) + y[k][j];
    }
  }
}

// `cp` holds each block's K BT NCH-word y slot; the launch brings 5 K BT NCH words of dynamic shared memory (the p fk
// slot, then the 4 trips' KK slots). Logical lane L = (blockIdx.x BT + tid) K + k reads row L mod 1024 of st and W.
template <int K, int BT, int WIDE>
__global__ void __launch_bounds__(BT) fbk(const uint32_t *st, const uint32_t *W, uint32_t *out, uint32_t *cp,
                                          long long *cyc, int sets, uint32_t one) {
  constexpr int SLOT = K * BT * NCH;
  extern __shared__ __align__(128) uint32_t smem[];
  const int tid = threadIdx.x, base = (blockIdx.x * BT + tid) * K;
  uint32_t *sd = smem, *sk = smem + SLOT;
  uint32_t y[K][NCH], z[K][NCH], d[K][8][4];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int src = (base + k) & 1023;
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      y[k][j] = st[src * 16 + j];
      z[k][j] = st[src * 16 + NCH + j];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) d[k][i][e] = W[src * 32 + 4 * i + e];
  }
  for (int i = tid; i < 4 * SLOT; i += BT) sk[i] = KK(2 * (i / SLOT) + (i % NCH) / 4);
  asm volatile("fence.proxy.async;" ::: "memory");
  uint32_t *gb = cp + (size_t)blockIdx.x * SLOT;
  const uint32_t bytes = SLOT * 4u;
  const uint32_t sdb = (uint32_t)__cvta_generic_to_shared(sd);
  const uint32_t skb = (uint32_t)__cvta_generic_to_shared(sk);
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int s = 0; s < sets; ++s) {
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int i = 0; i < 8; ++i) imma(d[k][i], one, one);
    trip<K, BT, WIDE, 0>(y, z, d, gb, sdb, skb, bytes, tid);
    trip<K, BT, WIDE, 1>(y, z, d, gb, sdb, skb + 4u * SLOT, bytes, tid);
    trip<K, BT, WIDE, 2>(y, z, d, gb, sdb, skb + 8u * SLOT, bytes, tid);
    trip<K, BT, WIDE, 3>(y, z, d, gb, sdb, skb + 12u * SLOT, bytes, tid);
  }
  __syncthreads();
  long long t1 = clock64();
#pragma unroll
  for (int k = 0; k < K; ++k)
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      out[(size_t)(base + k) * 16 + j] = y[k][j];
      out[(size_t)(base + k) * 16 + NCH + j] = z[k][j];
    }
  if (tid == 0) cyc[blockIdx.x] = t1 - t0;
}

__device__ __forceinline__ void imma6(uint32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                      uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// fbk's trip without the glue ptxas rematerializes there: the lane's global and shared slot addresses come in opaque
// (gt, st), each logical warp's an immediate offset from them, and the bulk body sits under a warp-uniform branch with
// elect.sync's predicate inside the asm, so no warp takes a divergent branch
template <int K, int BT, int H>
__device__ __forceinline__ void tripu(uint32_t (&y)[K][NCH], uint32_t (&z)[K][NCH], const uint32_t (&d)[K][8][4],
                                      uint32_t *gt, uint32_t st, uint32_t *gb, uint32_t sd, uint32_t sk, uint32_t bytes,
                                      int wid) {
#pragma unroll
  for (int k = 0; k < K; ++k) {
    uint32_t m[NCH];
#pragma unroll
    for (int j = 0; j < NCH; ++j) m[j] = d[k][2 * H + j / 4][j % 4] * FK(4 * (2 * H + j / 4) + j % 4);
    st256(gt + NCH * BT * k, y[k]);
    sts128(st + 4u * NCH * BT * k, m[0], m[1], m[2], m[3]);
    sts128(st + 4u * NCH * BT * k + 16, m[4], m[5], m[6], m[7]);
  }
  asm volatile("fence.proxy.async;" ::: "memory");
  __syncthreads();
  if (wid == 0)
    asm volatile("{\n\t.reg .pred p;\n\telect.sync _|p, 0xffffffff;\n\t"
                 "@p cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%1], %3;\n\t"
                 "@p cp.async.bulk.commit_group;\n\t@p cp.async.bulk.wait_group 0;\n\t"
                 "@p cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%2], %3;\n\t"
                 "@p cp.async.bulk.commit_group;\n\t@p cp.async.bulk.wait_group 0;\n\t"
                 "@p fence.proxy.async;\n\t}" ::"l"(gb), "r"(sd), "r"(sk), "r"(bytes)
                 : "memory");
  __syncthreads();
#pragma unroll
  for (int k = 0; k < K; ++k) {
    uint32_t u[NCH];
    ldcg256(gt + NCH * BT * k, u);
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      y[k][j] = u[j] * KA(2 * H + j / 4) + z[k][j];
      z[k][j] = __funnelshift_l(u[j], u[j], ROT(2 * H + j / 4)) + y[k][j];
    }
  }
}

// fbk at 256 bits with tripu: the IMMA's A and B fragments are six registers ptxas can't merge, so it holds them
// across the loop instead of reloading `one` and copying it each set
template <int K, int BT>
__global__ void __launch_bounds__(BT) fbu(const uint32_t *st, const uint32_t *W, uint32_t *out, uint32_t *cp,
                                          long long *cyc, int sets, uint32_t one) {
  constexpr int SLOT = K * BT * NCH;
  extern __shared__ __align__(128) uint32_t smem[];
  const int tid = threadIdx.x, base = (blockIdx.x * BT + tid) * K;
  const int wid = __shfl_sync(0xffffffffu, tid >> 5, 0);
  uint32_t *sd = smem, *sk = smem + SLOT;
  uint32_t y[K][NCH], z[K][NCH], d[K][8][4];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int src = (base + k) & 1023;
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      y[k][j] = st[src * 16 + j];
      z[k][j] = st[src * 16 + NCH + j];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) d[k][i][e] = W[src * 32 + 4 * i + e];
  }
  for (int i = tid; i < 4 * SLOT; i += BT) sk[i] = KK(2 * (i / SLOT) + (i % NCH) / 4);
  asm volatile("fence.proxy.async;" ::: "memory");
  uint32_t *gb = cp + (size_t)blockIdx.x * SLOT, *gt = gb + NCH * tid;
  const uint32_t bytes = SLOT * 4u;
  const uint32_t sdb = (uint32_t)__cvta_generic_to_shared(sd);
  const uint32_t skb = (uint32_t)__cvta_generic_to_shared(sk);
  uint32_t stt = sdb + 4u * NCH * tid;
  uint32_t a0 = one, a1 = one, a2 = one, a3 = one, b0 = one, b1 = one;
  asm volatile("" : "+l"(gt), "+r"(stt), "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(b0), "+r"(b1));
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int s = 0; s < sets; ++s) {
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int i = 0; i < 8; ++i) imma6(d[k][i], a0, a1, a2, a3, b0, b1);
    tripu<K, BT, 0>(y, z, d, gt, stt, gb, sdb, skb, bytes, wid);
    tripu<K, BT, 1>(y, z, d, gt, stt, gb, sdb, skb + 4u * SLOT, bytes, wid);
    tripu<K, BT, 2>(y, z, d, gt, stt, gb, sdb, skb + 8u * SLOT, bytes, wid);
    tripu<K, BT, 3>(y, z, d, gt, stt, gb, sdb, skb + 12u * SLOT, bytes, wid);
  }
  __syncthreads();
  long long t1 = clock64();
#pragma unroll
  for (int k = 0; k < K; ++k)
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      out[(size_t)(base + k) * 16 + j] = y[k][j];
      out[(size_t)(base + k) * 16 + NCH + j] = z[k][j];
    }
  if (tid == 0) cyc[blockIdx.x] = t1 - t0;
}

// the device pass sees no host code, so each variant is instantiated here
#define FBK_INST(K, BT, WIDE)                                                                                         \
  template __global__ void fbk<K, BT, WIDE>(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *,  \
                                            int, uint32_t);
FBK_INST(1, 512, 0)
FBK_INST(1, 512, 1)
FBK_INST(2, 256, 0)
FBK_INST(2, 256, 1)
FBK_INST(4, 128, 1)
template __global__ void fbu<2, 256>(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *, int,
                                     uint32_t);
template __global__ void fbu<4, 128>(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *, int,
                                     uint32_t);

#ifndef __CUDA_ARCH__
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

typedef void (*Kern)(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *, int, uint32_t);

static uint32_t h_rotl(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

static uint32_t st[1024 * 16], W[1024 * 32];

// n logical lanes through `sets` sets, each the pure fold: final states, lane-major, y then z
static void ref(int sets, int n, uint32_t *fin) {
  uint32_t(*S)[16] = (uint32_t(*)[16])fin;
  memcpy(fin, st, (size_t)n * 16 * sizeof(uint32_t));
  for (int s = 0; s < sets; ++s)
    for (int ln = 0; ln < n; ++ln) {
      uint32_t *y = S[ln], *z = S[ln] + NCH;
      for (int t = 0; t < 8; ++t)
        for (int e = 0; e < 4; ++e) {
          int c = 4 * (t % 2) + e;
          uint32_t p = W[ln * 32 + 4 * t + e] + 32u * (uint32_t)(s + 1);
          uint32_t u = y[c] ^ (p * FK(4 * t + e)) ^ KK(t);
          y[c] = u * KA(t) + z[c];
          z[c] = h_rotl(u, ROT(t)) + y[c];
        }
    }
}

int main() {
  int dev = 0, sms = 0, maj = 0, mnr = 0, clk = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev);
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
  printf("device cc=%d.%d sms=%d clock_khz=%d\n", maj, mnr, sms, clk);

  srand(0xC0B1);
  for (int i = 0; i < 1024 * 16; ++i) st[i] = (uint32_t)rand() * 668265263u ^ (uint32_t)rand();
  for (int i = 0; i < 1024 * 32; ++i) W[i] = (uint32_t)rand() * 2654435761u ^ (uint32_t)rand();
  const int BSETS = CPS, TSETS = 2048, LANES = 512;
  static uint32_t fin[LANES * 16];
  ref(BSETS, LANES, fin);

  struct { const char *name; Kern k; int K, bt, wide; } ks[] = {
      {"fbk_k1_128", fbk<1, 512, 0>, 1, 512, 0}, {"fbk_k1_256", fbk<1, 512, 1>, 1, 512, 1},
      {"fbk_k2_128", fbk<2, 256, 0>, 2, 256, 0}, {"fbk_k2_256", fbk<2, 256, 1>, 2, 256, 1},
      {"fbk_k4_256", fbk<4, 128, 1>, 4, 128, 1}, {"fbu_k2_256", fbu<2, 256>, 2, 256, 1},
      {"fbu_k4_256", fbu<4, 128>, 4, 128, 1},
  };
  const int smem = 5 * LANES * NCH * 4, maxblocks = sms * 32;
  uint32_t *d_st, *d_W, *d_out, *d_cp;
  long long *d_cyc;
  cudaMalloc(&d_st, sizeof st);
  cudaMalloc(&d_W, sizeof W);
  cudaMalloc(&d_out, (size_t)maxblocks * LANES * 16 * sizeof(uint32_t));
  cudaMalloc(&d_cp, (size_t)maxblocks * LANES * NCH * sizeof(uint32_t));
  cudaMalloc(&d_cyc, maxblocks * sizeof(long long));
  cudaMemcpy(d_st, st, sizeof st, cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, W, sizeof W, cudaMemcpyHostToDevice);
  long long *cyc = (long long *)malloc(maxblocks * sizeof(long long));
  const uint32_t one = 0x01010101u;
  int all = 1;
  for (auto &e : ks) {
    static uint32_t out[LANES * 16];
    cudaFuncSetAttribute((const void *)e.k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaMemset(d_cp, 0, LANES * NCH * sizeof(uint32_t));
    e.k<<<1, e.bt, smem>>>(d_st, d_W, d_out, d_cp, d_cyc, BSETS, one);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost);
    int bad = err != cudaSuccess;
    for (int i = 0; i < LANES * 16; ++i) bad += out[i] != fin[i];
    const int ok = bad == 0;
    all &= ok;
    printf("%s K=%d bt=%d wide=%d bitexact=%d logical_lanes=%d sets=%d bad_words=%d err=%s\n", e.name, e.K, e.bt,
           e.wide, ok, LANES, BSETS, bad, cudaGetErrorString(err));
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, e.k, e.bt, smem);
    for (int b = 1; b <= occ; b = b < occ ? occ : occ + 1) {
      int blocks = sms * b;
      cudaMemset(d_cp, 0, (size_t)blocks * LANES * NCH * sizeof(uint32_t));
      e.k<<<blocks, e.bt, smem>>>(d_st, d_W, d_out, d_cp, d_cyc, TSETS, one);
      err = cudaGetLastError();
      if (err == cudaSuccess) err = cudaDeviceSynchronize();
      cudaMemcpy(cyc, d_cyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
      long long mx = 0;
      for (int i = 0; i < blocks; ++i) mx = cyc[i] > mx ? cyc[i] : mx;
      const double wps = b * (e.bt / 32) / 4.0, units = 8.0 * e.K;
      const double cu = (double)mx / (wps * TSETS * units);
      printf("%s tp blocks_per_sm=%d occ=%d warps_per_smsp=%.2f smsp_cyc_per_unit=%.4f err=%s\n", e.name, b, occ, wps,
             cu, cudaGetErrorString(err));
    }
  }
  printf("all_bitexact=%d\n", all);
  return all ? 0 : 3;
}
#endif
