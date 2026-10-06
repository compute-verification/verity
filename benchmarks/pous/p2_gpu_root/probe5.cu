// Latencies that set the per-level cost of the floor model's trees, and one SM's int8 tensor-core rate on a dependent,
// squaring-sized loop. Device cycles (clock64), one CTA. Output: JSON lines.
//   chain  one thread, a dependent chain of one operation (FFMA, IMAD, IMAD.WIDE, IMAD.HI, an integer Shoup butterfly
//          level with Harvey's lazy reduction, an FP32 complex butterfly level in its fused two-FFMA form) or one warp
//          (SHFL): cycles per link.
//   smem   T threads pass a value through shared memory to the next warp and meet at __syncthreads, double-buffered:
//          cycles per pass. `bar` is the barrier alone; `ldchase` a one-thread dependent shared-memory load.
//   gemm   W warps run M mma.sync m16n8k32 s8 per round between barriers: every A and B fragment is read from shared
//          memory that the other warps wrote in the previous round, so each round depends on the last (as one
//          squaring does on the one before). Cycles per round, and MACs per cycle.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int L = 32;

__global__ void k_chain(int op, int n, uint32_t p0, uint32_t p1, unsigned long long *out, uint32_t *sink) {
  const uint32_t q = 1006632961u, w = p0 % q, wp = (uint32_t)(((uint64_t)w << 32) / q);
  float fx = __uint_as_float(0x3f800000u | (p0 & 0xffff)), fa = 0.999f + p1 * 1e-9f, fb = 1e-7f;
  float ar = fx, ai = fa, br = fb, bi = fx, wr = 0.70710678f, wi = 0.70710678f + p1 * 1e-9f;
  uint32_t u = p0 | 1, x = p0 % q, y = p1 % q;
  uint64_t v = p0;
  unsigned long long c0 = clock64(), c1;
  switch (op) {
  case 0:
    #pragma unroll 1
    for (int i = 0; i < n; i++) {
      #pragma unroll
      for (int j = 0; j < L; j++) fx = fmaf(fx, fa, fb);
    }
    break;
  case 1:
    #pragma unroll 1
    for (int i = 0; i < n; i++) {
      #pragma unroll
      for (int j = 0; j < L; j++) u = u * p1 + p0;
    }
    break;
  case 2:
    #pragma unroll 1
    for (int i = 0; i < n; i++) {
      #pragma unroll
      for (int j = 0; j < L; j++) v = (uint64_t)(uint32_t)v * p1 + (v >> 32);
    }
    break;
  case 3:
    #pragma unroll 1
    for (int i = 0; i < n; i++) {
      #pragma unroll
      for (int j = 0; j < L; j++) u = __umulhi(u, p1) + p0;
    }
    break;
  case 4:
    #pragma unroll 1
    for (int i = 0; i < n; i++) {
      #pragma unroll
      for (int j = 0; j < L; j++) {
        uint32_t t = __umulhi(y, wp), r = y * w - t * q;
        uint32_t nx = x + r, ny = x - r + 2 * q;
        x = nx >= 2 * q ? nx - 2 * q : nx;
        y = ny;
      }
    }
    break;
  case 5:
    #pragma unroll 1
    for (int i = 0; i < n; i++) {
      #pragma unroll
      for (int j = 0; j < L; j++) {
        float nar = fmaf(wr, br, fmaf(-wi, bi, ar)), nai = fmaf(wr, bi, fmaf(wi, br, ai));
        float nbr = fmaf(-wr, br, fmaf(wi, bi, ar)), nbi = fmaf(-wr, bi, fmaf(-wi, br, ai));
        ar = nar; ai = nai; br = nbr; bi = nbi;
      }
    }
    break;
  case 6: {
    // A lane-uniform op commutes with the shuffle and lets the compiler cancel pairs; a per-lane key does not.
    const uint32_t k = p0 * (threadIdx.x + 1);
    #pragma unroll 1
    for (int i = 0; i < n; i++) {
      #pragma unroll
      for (int j = 0; j < L; j++) u = __shfl_xor_sync(0xffffffffu, u ^ k, 1);
    }
    break;
  }
  }
  c1 = clock64();
  if (threadIdx.x == 0) out[0] = c1 - c0;
  uint32_t s = u ^ x ^ y ^ (uint32_t)v ^ __float_as_uint(fx) ^ __float_as_uint(ar + ai + br + bi);
  if (s == 0x12345678u) *sink = s;
}

__global__ void k_smem(int mode, int n, unsigned long long *out, uint32_t *sink) {
  __shared__ uint32_t s[2][1024];
  __shared__ uint32_t chase[1024];
  const int t = threadIdx.x, T = blockDim.x;
  s[0][t] = t; s[1][t] = t; chase[t] = (t * 97 + 1) & 1023;
  __syncthreads();
  uint32_t v = t;
  unsigned long long c0 = clock64();
  if (mode == 0) {
    #pragma unroll 1
    for (int i = 0; i < n; i++) {
      v = s[i & 1][(t + 32) % T] + 1;
      s[(i + 1) & 1][t] = v;
      __syncthreads();
    }
  } else if (mode == 1) {
    #pragma unroll 1
    for (int i = 0; i < n; i++) { v += i; __syncthreads(); }
  } else if (t == 0) {
    #pragma unroll 1
    for (int i = 0; i < n; i++) v = chase[v];
  }
  unsigned long long c1 = clock64();
  if (t == 0) out[0] = c1 - c0;
  if (v == 0xdeadbeefu) *sink = v;
}

constexpr int XW = 4096;

__global__ void k_gemm(int n, int per_warp, unsigned long long *out, uint32_t *sink) {
  __shared__ uint32_t X[2][XW];
  const int t = threadIdx.x, lane = t & 31, wid = t >> 5, T = blockDim.x;
  for (int i = t; i < XW; i += T) { X[0][i] = i * 2654435761u; X[1][i] = i; }
  __syncthreads();
  int d[4][4] = {};
  unsigned long long c0 = clock64();
  #pragma unroll 1
  for (int it = 0; it < n; it++) {
    const uint32_t *src = X[it & 1];
    uint32_t a[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) a[j] = src[(wid * 128 + j * 32 + lane) & (XW - 1)];
    #pragma unroll 1
    for (int m = 0; m < per_warp; m += 4) {
      #pragma unroll
      for (int q = 0; q < 4; q++) {
        if (m + q < per_warp) {
          const int bi = ((wid * 61 + (m + q) * 64) * 2 + lane * 2) & (XW - 2);
          mma_s8(d[q], a, src[bi], src[bi + 1]);
        }
      }
    }
    uint32_t r = 0;
    #pragma unroll
    for (int q = 0; q < 4; q++) r += d[q][0] ^ d[q][1] ^ d[q][2] ^ d[q][3];
    uint32_t *dst = X[(it + 1) & 1];
    for (int i = t; i < XW; i += T) dst[i] = r + i;
    __syncthreads();
  }
  unsigned long long c1 = clock64();
  if (t == 0) out[0] = c1 - c0;
  if (d[0][0] == 0x7fffffff) *sink = d[0][1];
}

int main() {
  unsigned long long *dout, h;
  uint32_t *sink;
  CK(cudaMalloc(&dout, 8)); CK(cudaMalloc(&sink, 4));
  const char *names[] = {"ffma", "imad", "imad_wide", "imad_hi", "bfly_shoup_u32", "bfly_fp32_complex", "shfl_xor"};
  const int n = 4096;
  for (int op = 0; op < 7; op++) {
    int th = op == 6 ? 32 : 1;
    for (int rep = 0; rep < 2; rep++) { k_chain<<<1, th>>>(op, n, 123457u, 987651u, dout, sink); CK(cudaDeviceSynchronize()); }
    CK(cudaMemcpy(&h, dout, 8, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"chain\",\"op\":\"%s\",\"cyc_per_link\":%.3f}\n", names[op], (double)h / ((double)n * L));
  }
  const char *sm[] = {"pass", "bar", "ldchase"};
  for (int mode = 0; mode < 3; mode++)
    for (int T : {64, 256, 512, 1024}) {
      if (mode == 2 && T != 64) continue;
      for (int rep = 0; rep < 2; rep++) { k_smem<<<1, T>>>(mode, n, dout, sink); CK(cudaDeviceSynchronize()); }
      CK(cudaMemcpy(&h, dout, 8, cudaMemcpyDeviceToHost));
      printf("{\"probe\":\"smem\",\"mode\":\"%s\",\"threads\":%d,\"cyc\":%.3f}\n", sm[mode], T, (double)h / n);
    }
  const int g = 512;
  for (int W : {4, 8, 16, 32})
    for (int M : {0, 64, 128, 256, 330, 512, 1024, 1343, 2048}) {
      int pw = (M + W - 1) / W;
      for (int rep = 0; rep < 2; rep++) { k_gemm<<<1, 32 * W>>>(g, pw, dout, sink); CK(cudaDeviceSynchronize()); }
      CK(cudaMemcpy(&h, dout, 8, cudaMemcpyDeviceToHost));
      double cyc = (double)h / g;
      printf("{\"probe\":\"gemm\",\"warps\":%d,\"mma_per_round\":%d,\"macs_per_round\":%d,\"cyc_per_round\":%.1f,"
             "\"macs_per_clk\":%.1f}\n", W, pw * W, pw * W * 4096, cyc, pw * W * 4096.0 / cyc);
    }
  return 0;
}
