// RT4-SHEET-1 (b): fbw's whole-block foldmem fold (fold_bulk.cu, 15.7656 a unit at DS_issue) made into an ACCEPTED
// unit by adding CP-W's per-set checkpoint (`PouwWarpCheck_v2`, decision 104): after every set the warp's two
// checkpoint words, the XOR over its 32 lanes and 8 chains of y and of z (`segment.warp_checkpoint`). fbw is
// FOLD-PROOF's open whole-block variant (#2891: mode 10 with the KK slot reloaded each trip); it folds below
// N_w phi* = 16 but, as built, emits no checkpoint, so it is not accepted. This asks what the checkpoint costs a unit
// on the whole-block fold, and so whether the 0.656% headline (break line 16.99993 a unit, #2791 Headline.lean) holds.
//
// The packing that makes the fold cheap (K = 4 logical warps a thread, so one body serves 2 K units a trip) scrambles
// the warp structure: a logical warp's 32 lanes are 8 physical threads x 4 lanes, so a logical warp != a physical warp.
// RT2-120's cheapest built checkpoint (pair2 / redux2, 0.25 a unit by egress, 1.0 issue-merged) reduces over the
// physical warp (REDUX, __reduce_xor_sync), which the packing breaks. The cheater's checkpoint over a logical warp must
// instead merge across physical threads; the cheapest built route here is a per-thread register XOR of its K NCH state
// words (the 8 threads' outputs combine on the memory system by red.global.xor into the zeroed receipt pair). `main`
// runs on the device it finds: bitexact over 1024 logical lanes and 64 sets (every final state AND every set's
// checkpoint against the host), then tp. fold_cp.sh prices each variant's SASS a unit; the checkpoint's cost a unit is
// that price minus fold_bulk.sh's fbw price for the same K, BT, KS.
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

__device__ __forceinline__ void imma6(uint32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void st256(uint32_t *p, const uint32_t (&v)[NCH]) {
  asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(p), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]),
               "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7])
               : "memory");
}
__device__ __forceinline__ void sts128(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void ldcg256(const uint32_t *p, uint32_t (&v)[NCH]) {
  asm volatile("ld.global.cg.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
               : "l"(p)
               : "memory");
}
__device__ __forceinline__ void bulk_xor_issue(uint32_t *g, uint32_t s, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%1], %2;" ::"l"(g), "r"(s), "r"(bytes)
               : "memory");
}
// the cross-thread merge on the memory system: the lane's XORed state word into the warp's zeroed receipt word
__device__ __forceinline__ void red_g_xor(uint32_t *g, uint32_t v) {
  asm volatile("red.global.relaxed.gpu.xor.b32 [%0], %1;" ::"l"(g), "r"(v) : "memory");
}

// fbw's trip (fold_bulk.cu tripw): the KK slab reloaded each trip, the fold's XOR on the bulk engine
template <int K, int BT, int KS, int H>
__device__ __forceinline__ void tripw(uint32_t (&y)[K][NCH], uint32_t (&z)[K][NCH], const uint32_t (&d)[K][8][4],
                                      uint32_t *gt, uint32_t st, uint32_t *gb, uint32_t sd, uint32_t sk, int wid,
                                      int ln) {
  constexpr int SLOT = K * BT * NCH;
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
  if (wid == 0 && ln == 0) {
    bulk_xor_issue(gb, sd, SLOT * 4u);
#pragma unroll
    for (int c = 0; c < SLOT / KS; ++c) bulk_xor_issue(gb + c * KS, sk, KS * 4u);
    asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group 0;\n\tfence.proxy.async;" ::: "memory");
  }
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

// fbw + CP-W. `cpk` holds the block's checkpoints, set-major then logical warp then (y, z); it is zeroed by the host.
template <int K, int BT, int KS>
__global__ void __launch_bounds__(BT) fcp(const uint32_t *st, const uint32_t *W, uint32_t *out, uint32_t *cp,
                                         uint32_t *cpk, long long *cyc, int sets, uint32_t one) {
  constexpr int SLOT = K * BT * NCH;
  static_assert(SLOT % KS == 0 && KS % NCH == 0, "the KK slab tiles the slot");
  const int NWARP = (gridDim.x * BT * K) / 32;
  extern __shared__ __align__(128) uint32_t smem[];
  const int tid = threadIdx.x, base = (blockIdx.x * BT + tid) * K;
  const int wid = __shfl_sync(0xffffffffu, tid >> 5, 0), ln = tid & 31;
  const int lwarp = base / 32;                      // this thread's logical warp (its K lanes share it)
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
  for (int i = tid; i < 4 * KS; i += BT) sk[i] = KK(2 * (i / KS) + (i % NCH) / 4);
  asm volatile("fence.proxy.async;" ::: "memory");
  uint32_t *gb = cp + (size_t)blockIdx.x * SLOT, *gt = gb + NCH * tid;
  const uint32_t sdb = (uint32_t)__cvta_generic_to_shared(sd);
  const uint32_t skb = (uint32_t)__cvta_generic_to_shared(sk);
  uint32_t stt = sdb + 4u * NCH * tid;
  uint32_t a0 = one, a1 = one, a2 = one, a3 = one, b0 = one, b1 = one;
  uint32_t *r = cpk + 2 * lwarp;                    // set s's receipt pair is r + 2 NWARP s
  asm volatile("" : "+l"(gt), "+r"(stt), "+l"(r), "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(b0), "+r"(b1));
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int s = 0; s < sets; ++s) {
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int i = 0; i < 8; ++i) imma6(d[k][i], a0, a1, a2, a3, b0, b1);
    tripw<K, BT, KS, 0>(y, z, d, gt, stt, gb, sdb, skb, wid, ln);
    tripw<K, BT, KS, 1>(y, z, d, gt, stt, gb, sdb, skb + 4u * KS, wid, ln);
    tripw<K, BT, KS, 2>(y, z, d, gt, stt, gb, sdb, skb + 8u * KS, wid, ln);
    tripw<K, BT, KS, 3>(y, z, d, gt, stt, gb, sdb, skb + 12u * KS, wid, ln);
    // CP-W: the lane's eight chains merged in registers, the 32 lanes merged on the memory system
    uint32_t cy = 0, cz = 0;
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < NCH; ++j) {
        cy ^= y[k][j];
        cz ^= z[k][j];
      }
    red_g_xor(r, cy);
    red_g_xor(r + 1, cz);
    r += 2 * NWARP;
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

#define FCP_INST(K, BT, KS)                                                                                          \
  template __global__ void fcp<K, BT, KS>(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, uint32_t *,    \
                                          long long *, int, uint32_t);
FCP_INST(4, 128, 4096)
FCP_INST(4, 256, 4096)
FCP_INST(4, 256, 2048)

#ifndef __CUDA_ARCH__
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

typedef void (*Kern)(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, uint32_t *, long long *, int,
                     uint32_t);

static uint32_t h_rotl(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }
static uint32_t st[1024 * 16], W[1024 * 32];

// n logical lanes through `sets` sets: final states (lane-major, y then z) in `fin`, and after each set the per-logical
// warp checkpoint (y, z) = XOR over 32 lanes and 8 chains, in `cpk` (set-major, then logical warp, then y, z)
static void ref(int sets, int n, uint32_t *fin, uint32_t *cpk) {
  uint32_t(*S)[16] = (uint32_t(*)[16])fin;
  const int NWARP = n / 32;
  memcpy(fin, st, (size_t)n * 16 * sizeof(uint32_t));
  for (int s = 0; s < sets; ++s) {
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
    for (int w = 0; w < NWARP; ++w) {
      uint32_t cy = 0, cz = 0;
      for (int ln = 32 * w; ln < 32 * (w + 1); ++ln)
        for (int j = 0; j < NCH; ++j) {
          cy ^= S[ln][j];
          cz ^= S[ln][NCH + j];
        }
      cpk[(s * NWARP + w) * 2 + 0] = cy;
      cpk[(s * NWARP + w) * 2 + 1] = cz;
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
  const int BSETS = CPS, TSETS = 2048, LANES = 1024, NWARP = LANES / 32;
  static uint32_t fin[LANES * 16], cpref[CPS * (LANES / 32) * 2];
  ref(BSETS, LANES, fin, cpref);

  struct { const char *name; Kern k; int K, bt, smem; } ks[] = {
      {"fcp4096_K4_BT128", fcp<4, 128, 4096>, 4, 128, (4096 + 4 * 4096) * 4},
      {"fcp4096_K4_BT256", fcp<4, 256, 4096>, 4, 256, (8192 + 4 * 4096) * 4},
      {"fcp2048_K4_BT256", fcp<4, 256, 2048>, 4, 256, (8192 + 4 * 2048) * 4},
  };
  const int maxblocks = sms * 32;
  uint32_t *d_st, *d_W, *d_out, *d_cp, *d_cpk;
  long long *d_cyc;
  cudaMalloc(&d_st, sizeof st);
  cudaMalloc(&d_W, sizeof W);
  cudaMalloc(&d_out, (size_t)maxblocks * LANES * 16 * sizeof(uint32_t));
  cudaMalloc(&d_cp, (size_t)maxblocks * LANES * NCH * sizeof(uint32_t));
  // the throughput pass writes TSETS sets of every block's logical warps (at most 32 a block); these variants'
  // shared memory (80-96 KB) allows one block a SM, so sms * 4 blocks is room to spare
  const size_t cpk_words = (size_t)TSETS * sms * 4 * 32 * 2;
  cudaMalloc(&d_cpk, cpk_words * sizeof(uint32_t));
  cudaMalloc(&d_cyc, maxblocks * sizeof(long long));
  cudaMemcpy(d_st, st, sizeof st, cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, W, sizeof W, cudaMemcpyHostToDevice);
  long long *cyc = (long long *)malloc(maxblocks * sizeof(long long));
  const uint32_t one = 0x01010101u;
  int all = 1;
  for (auto &e : ks) {
    static uint32_t out[LANES * 16], cko[CPS * (LANES / 32) * 2];
    const int smem = e.smem, lanes = e.K * e.bt, nwarp = lanes / 32;
    cudaFuncSetAttribute((const void *)e.k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaMemset(d_cp, 0, LANES * NCH * sizeof(uint32_t));
    cudaMemset(d_cpk, 0, (size_t)BSETS * nwarp * 2 * sizeof(uint32_t));
    e.k<<<1, e.bt, smem>>>(d_st, d_W, d_out, d_cp, d_cpk, d_cyc, BSETS, one);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, (size_t)lanes * 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cko, d_cpk, (size_t)BSETS * nwarp * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    int bad_st = err != cudaSuccess, bad_cp = 0;
    for (int i = 0; i < lanes * 16; ++i) bad_st += out[i] != fin[i];
    // one block's logical warps are warps 0..nwarp-1 of the reference's lane-major checkpoints
    for (int s = 0; s < BSETS; ++s)
      for (int w = 0; w < nwarp; ++w)
        for (int j = 0; j < 2; ++j) bad_cp += cko[(s * nwarp + w) * 2 + j] != cpref[(s * NWARP + w) * 2 + j];
    const int ok = bad_st == 0 && bad_cp == 0;
    all &= ok;
    printf("%s K=%d bt=%d smem=%d bitexact=%d logical_lanes=%d sets=%d bad_states=%d bad_checkpoints=%d err=%s\n",
           e.name, e.K, e.bt, smem, ok, lanes, BSETS, bad_st, bad_cp, cudaGetErrorString(err));
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, e.k, e.bt, smem);
    for (int b = 1; b <= occ && b <= 4; b = b < occ ? occ : occ + 1) {
      int blocks = sms * b;
      cudaMemset(d_cp, 0, (size_t)blocks * LANES * NCH * sizeof(uint32_t));
      cudaMemset(d_cpk, 0, (size_t)blocks * CPS * nwarp * 2 * sizeof(uint32_t));
      e.k<<<blocks, e.bt, smem>>>(d_st, d_W, d_out, d_cp, d_cpk, d_cyc, TSETS, one);
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
