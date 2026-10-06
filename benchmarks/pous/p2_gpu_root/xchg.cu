// The exchange a cluster squaring chain does every step, alone: each of C CTAs owns 1/C of a 633-word vector and must end
// the round with the whole vector in its shared memory. Rounds are dependent (round t+1's data is sent only once round t's
// has arrived), so the time per round is the exchange's latency, the floor of any design that redistributes the value
// every squaring. Variants:
//   0  st.shared::cluster u32 to every CTA, then barrier.cluster
//   1  st.async u32, each completing bytes on the receiver's mbarrier; wait on the local mbarrier
//   2  st.async v4.u32 (16-byte), same
//   3  cp.async.bulk shared::cta -> shared::cluster, one bulk copy per destination, same mbarrier
//   4  tagged words: each word carries the round in its top 5 bits (a digit needs 27), st.shared::cluster u32, and the
//      receiver polls its own buffer until every word has the round's tag, then __syncthreads
//   5  tagged words, st.shared::cluster v4.u32
//   6  tagged words, st.shared::cluster v4.u32, only warp 0 polls (the others wait at __syncthreads)
// The vector length nd sweeps from 4 C words (one 16-byte store per CTA pair) to 636, to separate latency from volume.
// Output: one JSON line per (C, variant): device cycles and ns per round (CTA 0's clock64 and %globaltimer).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static const int ND = 636;   // 633 rounded up to a multiple of 4 words

__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t su32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t r) { uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o; }
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n .reg .pred p;\n W%=: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra W%=;\n}" :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}

template <int C, int V>
__global__ void xchg(int rounds, int nd, unsigned long long *out) {
  __shared__ __align__(16) uint32_t buf[2][ND];
  __shared__ __align__(16) uint32_t mine2[2][ND];
  __shared__ __align__(8) uint64_t mbar[2];
  cg::cluster_group cl = cg::this_cluster();
  const int g = cl.block_rank(), tid = threadIdx.x, tpb = blockDim.x;
  const int per = nd / C / 4 * 4 + (nd / C % 4 ? 4 : 0);
  const int a = min(nd, g * per), n = min(nd, a + per) - a;
  for (int q = tid; q < ND; q += tpb) { buf[0][q] = buf[1][q] = q; mine2[0][q] = mine2[1][q] = q; }
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su32(&mbar[0])));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su32(&mbar[1])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cl.sync();
  if (V > 0 && V < 4 && tid == 0) expect(su32(&mbar[1]), nd * 4);
  uint64_t c0 = clock64(), g0 = gtime();
  for (int t = 0; t < rounds; t++) {
    const int b = (t + 1) & 1;
    uint32_t *cur = buf[t & 1], *nxt = buf[b], *mine = mine2[t & 1];
    // this round's data depends on the last round's: own slice = cur's slice + 1
    if (V >= 4) {
      const uint32_t tag = (uint32_t)((t + 1) & 31) << 27;
      for (int q = tid; q < n; q += tpb) mine[a + q] = (((cur[a + q] & 0x7ffffffu) + 1) & 0x7ffffffu) | tag;
    } else {
      for (int q = tid; q < n; q += tpb) mine[a + q] = cur[a + q] + 1;
    }
    __syncthreads();
    if (V >= 4) {
      const uint32_t tag = (uint32_t)((t + 1) & 31);
      if (V == 4) {
        for (int q = tid; q < C * n; q += tpb) {
          int h = q / n, j = q - h * n;
          asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(mapa(su32(&nxt[a + j]), h)), "r"(mine[a + j]) : "memory");
        }
      } else {
        const int n4 = n / 4;
        for (int q = tid; q < C * n4; q += tpb) {
          int h = q / n4, j = 4 * (q - h * n4);
          uint4 v = *(const uint4 *)&mine[a + j];
          asm volatile("st.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(mapa(su32(&nxt[a + j]), h)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
        }
      }
      for (int q = tid; q < nd && (V != 6 || tid < 32); q += (V == 6 ? 32 : tpb)) {
        volatile uint32_t *w = &nxt[q];
        while ((*w >> 27) != tag) {}
      }
      __syncthreads();
    } else if (V == 0) {
      for (int q = tid; q < C * n; q += tpb) {
        int h = q / n, j = q - h * n;
        cl.map_shared_rank(nxt, h)[a + j] = mine[a + j];
      }
      cl.sync();
    } else {
      const uint32_t lbar = su32(&mbar[b]);
      if (V == 1) {
        for (int q = tid; q < C * n; q += tpb) {
          int h = q / n, j = q - h * n;
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];"
                       :: "r"(mapa(su32(&nxt[a + j]), h)), "r"(mine[a + j]), "r"(mapa(lbar, h)) : "memory");
        }
      } else if (V == 2) {
        const int n4 = n / 4;
        for (int q = tid; q < C * n4; q += tpb) {
          int h = q / n4, j = 4 * (q - h * n4);
          uint4 v = *(const uint4 *)&mine[a + j];
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.u32 [%0], {%1, %2, %3, %4}, [%5];"
                       :: "r"(mapa(su32(&nxt[a + j]), h)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(mapa(lbar, h)) : "memory");
        }
      } else {
        if (tid < C) {
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(mapa(su32(&nxt[a]), tid)), "r"(su32(&mine[a])), "r"(n * 4), "r"(mapa(lbar, tid)) : "memory");
        }
      }
      mbar_wait(lbar, (t >> 1) & 1);
      if (tid == 0 && t + 1 < rounds) expect(su32(&mbar[b ^ 1]), nd * 4);
    }
  }
  uint64_t c1 = clock64(), g1 = gtime();
  cl.sync();
  if (g == 0 && tid == 0) { out[0] = c1 - c0; out[1] = g1 - g0; out[2] = buf[rounds & 1][5]; }
}

template <int C, int V>
static void run(int tpb, int rounds, int nd, unsigned long long *d) {
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(C); cfg.blockDim = dim3(tpb); cfg.attrs = at; cfg.numAttrs = 1;
  if (C > 8) CK(cudaFuncSetAttribute(xchg<C, V>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  for (int w = 0; w < 2; w++) { CK(cudaLaunchKernelEx(&cfg, xchg<C, V>, rounds, nd, d)); CK(cudaDeviceSynchronize()); }
  unsigned long long h[3]; CK(cudaMemcpy(h, d, 24, cudaMemcpyDeviceToHost));
  const char *nm[] = {"st.cluster+barrier", "st.async.u32+mbarrier", "st.async.v4+mbarrier", "cp.async.bulk+mbarrier", "tagged-st.u32+poll", "tagged-st.v4+poll", "tagged-st.v4+warp0-poll"};
  printf("{\"probe\":\"exchange\",\"C\":%d,\"variant\":\"%s\",\"tpb\":%d,\"nd\":%d,\"rounds\":%d,\"cyc_per_round\":%.1f,\"ns_per_round\":%.2f,\"check\":%s}\n",
         C, nm[V], tpb, nd, rounds, (double)h[0] / rounds, (double)h[1] / rounds, (V >= 4 ? (h[2] & 0x7ffffffull) : h[2]) == 5ull + rounds ? "true" : "false");
}

// L2 exchange: C CTAs (a plain grid, one per SM, all resident), the vector double-buffered in global memory as tagged
// words; CTA g writes its slice with 16-byte relaxed.gpu stores, every CTA polls the whole vector with 16-byte relaxed.gpu
// loads (served by L2) until each word has the round's tag, and keeps a stripped copy in shared memory.
template <int C>
__global__ void xchg_l2(int rounds, int nd, uint32_t *G, unsigned long long *out) {
  __shared__ __align__(16) uint32_t cur[ND];
  const int g = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
  const int per = nd / C / 4 * 4 + (nd / C % 4 ? 4 : 0);
  const int a = min(nd, g * per), n = min(nd, a + per) - a;
  for (int q = tid; q < ND; q += tpb) cur[q] = q;
  __syncthreads();
  uint64_t c0 = clock64(), g0 = gtime();
  for (int t = 0; t < rounds; t++) {
    const uint32_t tag = (uint32_t)(t % 31 + 1) << 27;
    uint32_t *nx = G + ((t + 1) & 1) * ND;
    for (int q = 4 * tid; q < n; q += 4 * tpb) {
      uint4 v = *(const uint4 *)&cur[a + q];
      v.x = ((v.x + 1) & 0x7ffffffu) | tag; v.y = ((v.y + 1) & 0x7ffffffu) | tag;
      v.z = ((v.z + 1) & 0x7ffffffu) | tag; v.w = ((v.w + 1) & 0x7ffffffu) | tag;
      asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(nx + a + q), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    }
    __syncthreads();
    for (int q = 4 * tid; q < nd; q += 4 * tpb) {
      uint4 v;
      do {
        asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(nx + q) : "memory");
      } while ((v.x & ~0x7ffffffu) != tag || (v.y & ~0x7ffffffu) != tag || (v.z & ~0x7ffffffu) != tag || (v.w & ~0x7ffffffu) != tag);
      *(uint4 *)&cur[q] = make_uint4(v.x & 0x7ffffffu, v.y & 0x7ffffffu, v.z & 0x7ffffffu, v.w & 0x7ffffffu);
    }
    __syncthreads();
  }
  uint64_t c1 = clock64(), g1 = gtime();
  if (g == 0 && tid == 0) { out[0] = c1 - c0; out[1] = g1 - g0; out[2] = cur[5]; }
}

template <int C>
static void run_l2(int tpb, int rounds, int nd, uint32_t *G, unsigned long long *d) {
  for (int w = 0; w < 2; w++) {
    CK(cudaMemset(G, 0, 2 * ND * 4));
    xchg_l2<C><<<C, tpb>>>(rounds, nd, G, d); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  }
  unsigned long long h[3]; CK(cudaMemcpy(h, d, 24, cudaMemcpyDeviceToHost));
  printf("{\"probe\":\"exchange\",\"C\":%d,\"variant\":\"l2-tagged-v4-poll\",\"tpb\":%d,\"nd\":%d,\"rounds\":%d,\"cyc_per_round\":%.1f,\"ns_per_round\":%.2f,\"check\":%s}\n",
         C, tpb, nd, rounds, (double)h[0] / rounds, (double)h[1] / rounds, h[2] == 5ull + rounds ? "true" : "false");
}

template <int C> static void all_l2(int tpb, int rounds, uint32_t *G, unsigned long long *d) {
  for (int nd : {4 * C, 64 > 4 * C ? 64 : 4 * C, 160, 320, ND}) if (nd <= ND) run_l2<C>(tpb, rounds, nd, G, d);
}

template <int C> static void all(int tpb, int rounds, unsigned long long *d) {
  for (int nd : {4 * C, 64 > 4 * C ? 64 : 4 * C, 160, 320, ND}) {
    run<C, 0>(tpb, rounds, nd, d); run<C, 1>(tpb, rounds, nd, d); run<C, 2>(tpb, rounds, nd, d); run<C, 3>(tpb, rounds, nd, d);
    run<C, 4>(tpb, rounds, nd, d); run<C, 5>(tpb, rounds, nd, d); run<C, 6>(tpb, rounds, nd, d);
  }
}

int main(int argc, char **argv) {
  unsigned long long *d; CK(cudaMalloc(&d, 64));
  const int rounds = 20000;
  const int tpb = argc > 1 ? atoi(argv[1]) : 256;
  const bool dsmem = argc > 2 ? atoi(argv[2]) != 0 : true;
  if (dsmem) { all<2>(tpb, rounds, d); all<4>(tpb, rounds, d); all<8>(tpb, rounds, d); all<16>(tpb, rounds, d); }
  uint32_t *G; CK(cudaMalloc(&G, 2 * ND * 4));
  all_l2<2>(tpb, rounds, G, d); all_l2<4>(tpb, rounds, G, d); all_l2<8>(tpb, rounds, G, d); all_l2<16>(tpb, rounds, G, d);
  all_l2<32>(tpb, rounds, G, d);   // 64 CTAs polling one 160-word vector without backoff stalled for minutes
  return 0;
}
