// R4 (CMP-12 red team of M4): do memory-side adds keep their rate while the tensor cores are saturated and L2 is
// streaming? Three warp groups in one kernel, two CTAs an SM, any subset active (mode bits):
//   T (bit 0): 4 warps a CTA, 8 independent mma.sync.m16n8k32 E4M3 chains each, register-resident;
//   R (bit 1): 2 warps a CTA, red.global.add.f32 (atomicAdd, result unused) into a 32 MiB L2-resident buffer;
//   L (bit 2): 2 warps a CTA, ld.global.cg.v4 streaming a 64 MiB L2-resident buffer.
// Each group's solo run is calibrated to about 150 ms; every mode then prints each active group's rate, per SM per
// clock, over that group's own span (first start to last end, globaltimer), best of 3.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

constexpr unsigned NRED = 8u << 20;  // floats: 32 MiB
constexpr unsigned NLD4 = 4u << 20;  // float4s: 64 MiB

struct Slots {
  unsigned long long t0[3], t1[3], clk[3], ns[3];
  float sink;
};

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void __launch_bounds__(256) probe(int mode, long long nt, long long nr, long long nl, float* rbuf,
                                             const float4* lbuf, Slots* s) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int g = warp < 4 ? 0 : (warp < 6 ? 1 : 2);
  if (!(mode & (1 << g))) return;
  unsigned long long t0 = gtime();
  long long c0 = clock64();
  float acc = 0.f;
  if (g == 0) {
    float d[8][4];
#pragma unroll
    for (int j = 0; j < 8; j++) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f;
    unsigned a0 = 0x38383838u, a1 = 0x30303030u, a2 = 0x38303830u ^ (unsigned)lane, a3 = 0x30383038u;
    unsigned b0 = 0x38383838u, b1 = 0x30303030u;
    for (long long i = 0; i < nt; i++) {
#pragma unroll
      for (int j = 0; j < 8; j++)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
#pragma unroll
    for (int j = 0; j < 8; j++) acc += d[j][0] + d[j][1] + d[j][2] + d[j][3];
  } else if (g == 1) {
    unsigned tid = (blockIdx.x * 2 + (warp - 4)) * 32 + lane, nth = gridDim.x * 64;
    for (long long i = 0; i < nr; i++) atomicAdd(&rbuf[(tid + (unsigned)i * nth) & (NRED - 1)], 1.0f);
  } else {
    unsigned tid = (blockIdx.x * 2 + (warp - 6)) * 32 + lane, nth = gridDim.x * 64;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (long long i = 0; i < nl; i += 4) {
#pragma unroll
      for (int u = 0; u < 4; u++) {
        float4 x = __ldcg(&lbuf[(tid + (unsigned)(i + u) * nth) & (NLD4 - 1)]);
        v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
      }
    }
    acc = v.x + v.y + v.z + v.w;
  }
  long long c1 = clock64();
  unsigned long long t1 = gtime();
  if (lane == 0) {
    atomicMin(&s->t0[g], t0);
    atomicMax(&s->t1[g], t1);
    if (blockIdx.x == 0) {
      atomicMax(&s->clk[g], (unsigned long long)(c1 - c0));
      atomicMax(&s->ns[g], t1 - t0);
    }
  }
  if (acc == 12345.678f) s->sink = acc;
}

static int sms;
static float* rbuf;
static float4* lbuf;
static Slots* dslots;

// Runs one mode; fills span (ns) and MHz per group.
static int run(int mode, const long long n[3], double span[3], double mhz[3]) {
  Slots h;
  for (int g = 0; g < 3; g++) { h.t0[g] = ~0ull; h.t1[g] = 0; h.clk[g] = 0; h.ns[g] = 0; }
  h.sink = 0;
  CK(cudaMemcpy(dslots, &h, sizeof h, cudaMemcpyHostToDevice));
  probe<<<sms * 2, 256>>>(mode, n[0], n[1], n[2], rbuf, lbuf, dslots);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&h, dslots, sizeof h, cudaMemcpyDeviceToHost));
  for (int g = 0; g < 3; g++) {
    span[g] = (mode >> g & 1) ? (double)(h.t1[g] - h.t0[g]) : 0;
    mhz[g] = (mode >> g & 1) && h.ns[g] ? 1e3 * (double)h.clk[g] / (double)h.ns[g] : 0;
  }
  return 0;
}

static double ops(int g, long long n) {
  double ctas = sms * 2.0;
  if (g == 0) return ctas * 4 * 8 * 4096.0 * n;  // MACs
  if (g == 1) return ctas * 64 * (double)n;      // FP32 adds
  return ctas * 64 * 16.0 * n;                   // bytes
}

int main() {
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaMalloc(&rbuf, (size_t)NRED * 4));
  CK(cudaMalloc(&lbuf, (size_t)NLD4 * 16));
  CK(cudaMalloc(&dslots, sizeof(Slots)));
  CK(cudaMemset(rbuf, 0, (size_t)NRED * 4));
  CK(cudaMemset(lbuf, 0, (size_t)NLD4 * 16));
  const char* name[3] = {"tensor_mac", "red_add", "l2_load_bytes"};
  long long n[3] = {2000, 20000, 20000};
  double span[3], mhz[3];
  // Calibrate each group alone to about 150 ms.
  for (int g = 0; g < 3; g++) {
    if (run(1 << g, n, span, mhz)) return 1;
    for (int k = 0; k < 2; k++) {
      n[g] = (long long)((double)n[g] * 150e6 / (span[g] > 1 ? span[g] : 1));
      n[g] = (n[g] + 3) / 4 * 4;
      if (run(1 << g, n, span, mhz)) return 1;
    }
  }
  printf("{\"probe\":\"r4_tc_red\",\"sms\":%d,\"n\":[%lld,%lld,%lld]}\n", sms, n[0], n[1], n[2]);
  for (int mode = 1; mode < 8; mode++) {
    double best[3] = {1e30, 1e30, 1e30}, bm[3] = {0, 0, 0};
    for (int rep = 0; rep < 3; rep++) {
      if (run(mode, n, span, mhz)) return 1;
      for (int g = 0; g < 3; g++)
        if ((mode >> g & 1) && span[g] < best[g]) { best[g] = span[g]; bm[g] = mhz[g]; }
    }
    for (int g = 0; g < 3; g++) {
      if (!(mode >> g & 1)) continue;
      double rate = ops(g, n[g]) / (best[g] * 1e-9);
      printf("{\"probe\":\"r4_tc_red\",\"mode\":\"%s%s%s\",\"group\":\"%s\",\"span_ms\":%.3f,\"mhz\":%.1f,"
             "\"per_s\":%.4e,\"per_sm_clk\":%.4f}\n",
             (mode & 1) ? "T" : "", (mode & 2) ? "R" : "", (mode & 4) ? "L" : "", name[g], best[g] * 1e-6, bm[g],
             rate, rate / sms / (bm[g] * 1e6));
    }
  }
  return 0;
}
