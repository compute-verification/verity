// Axiom probe: does an mma.sync accumulator register feed the next mma.sync as an operand with no instruction between,
// and with what semantics? Two paths on sm_120a, one warp each:
//   fp8: f32 accumulator bits read as four E4M3 operand bytes (decode(byte) enters the product exactly);
//   s8:  s32 accumulator bits read as four s8 operand bytes (byte 0 is v itself when |v| <= 127, v mod 256 otherwise).
// The second MMA's B selects one k per output column, so D2[i][j] = decode(A2[i][sel(j)]) exactly, checked on the host.
// Prints one JSON line per path. SASS for the kernels is checked by the driver script (no F2FP/I2F/F2I between QMMAs).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

__device__ __forceinline__ void mma_e4m3(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%10,%11,%12,%13};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void mma_s8(int d[4], const uint32_t a[4], const uint32_t b[2], const int c[4]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%10,%11,%12,%13};\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                 "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// a1/b1/c1: first MMA's fragments per lane; b2: the selection B per pass; d1o: first MMA's raw output bits;
// d2o: second MMA's output, whose A is d1's registers as they are.
__global__ void k_fp8(const uint32_t* a1, const uint32_t* b1, const float* c1, const uint32_t* b2, int npass,
                      uint32_t* d1o, float* d2o) {
  int l = threadIdx.x;
  uint32_t a[4], b[2];
  float c[4], d[4];
  for (int r = 0; r < 4; r++) { a[r] = a1[l * 4 + r]; c[r] = c1[l * 4 + r]; }
  b[0] = b1[l * 2]; b[1] = b1[l * 2 + 1];
  mma_e4m3(d, a, b, c);
  for (int r = 0; r < 4; r++) d1o[l * 4 + r] = __float_as_uint(d[r]);
  for (int p = 0; p < npass; p++) {
    uint32_t a2[4];
    for (int r = 0; r < 4; r++) a2[r] = __float_as_uint(d[r]);
    uint32_t bb[2] = {b2[(p * 32 + l) * 2], b2[(p * 32 + l) * 2 + 1]};
    float z[4] = {0.f, 0.f, 0.f, 0.f}, e[4];
    mma_e4m3(e, a2, bb, z);
    for (int r = 0; r < 4; r++) d2o[(p * 32 + l) * 4 + r] = e[r];
  }
}

__global__ void k_s8(const uint32_t* a1, const uint32_t* b1, const int* c1, const uint32_t* b2, int npass,
                     uint32_t* d1o, int* d2o) {
  int l = threadIdx.x;
  uint32_t a[4], b[2];
  int c[4], d[4];
  for (int r = 0; r < 4; r++) { a[r] = a1[l * 4 + r]; c[r] = c1[l * 4 + r]; }
  b[0] = b1[l * 2]; b[1] = b1[l * 2 + 1];
  mma_s8(d, a, b, c);
  for (int r = 0; r < 4; r++) d1o[l * 4 + r] = (uint32_t)d[r];
  for (int p = 0; p < npass; p++) {
    uint32_t a2[4];
    for (int r = 0; r < 4; r++) a2[r] = (uint32_t)d[r];
    uint32_t bb[2] = {b2[(p * 32 + l) * 2], b2[(p * 32 + l) * 2 + 1]};
    int z[4] = {0, 0, 0, 0}, e[4];
    mma_s8(e, a2, bb, z);
    for (int r = 0; r < 4; r++) d2o[(p * 32 + l) * 4 + r] = e[r];
  }
}

static double e4m3(uint8_t b) {
  int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  if (e == 15 && m == 7) return NAN;
  double v = e == 0 ? ldexp(m / 8.0, -6) : ldexp(1.0 + m / 8.0, e - 7);
  return s ? -v : v;
}
static uint8_t e4m3_of(double v) {
  for (int b = 0; b < 256; b++) if (e4m3((uint8_t)b) == v) return (uint8_t)b;
  return 0x7f;
}

// Element (i, k) of A from lane registers: lane (i%8)*4 + (k%16)/4, register (i>=8) + 2*(k>=16), byte k%4.
static uint8_t a_elem(const uint32_t* regs, int i, int k) {
  int lane = (i % 8) * 4 + (k % 16) / 4, r = (i >= 8) + 2 * (k >= 16);
  return (uint8_t)(regs[lane * 4 + r] >> (8 * (k % 4)));
}
static int sel(int p, int j) { return p * 8 + j; }

// B fragment for the selection matrix of pass p: column j has `one` at k = sel(p, j), zero elsewhere.
static void sel_b(uint32_t* b2, int npass, uint8_t one) {
  for (int p = 0; p < npass; p++)
    for (int l = 0; l < 32; l++) {
      int g = l / 4, t = l % 4;
      for (int h = 0; h < 2; h++) {
        uint32_t w = 0;
        for (int i = 0; i < 4; i++) { int k = 16 * h + 4 * t + i; if (k == sel(p, g)) w |= (uint32_t)one << (8 * i); }
        b2[(p * 32 + l) * 2 + h] = w;
      }
    }
}

int main() {
  const int NP = 4;
  std::mt19937_64 rng(20261015);
  uint32_t *da1, *db1, *db2, *dd1;
  float *dc1f, *dd2f;
  int *dc1i, *dd2i;
  CK(cudaMalloc(&da1, 128 * 4)); CK(cudaMalloc(&db1, 64 * 4)); CK(cudaMalloc(&db2, NP * 64 * 4));
  CK(cudaMalloc(&dd1, 128 * 4)); CK(cudaMalloc(&dc1f, 128 * 4)); CK(cudaMalloc(&dd2f, NP * 128 * 4));
  CK(cudaMalloc(&dc1i, 128 * 4)); CK(cudaMalloc(&dd2i, NP * 128 * 4));
  uint32_t a1[128], b1[64], b2[NP * 64], d1[128];
  float c1f[128], d2f[NP * 128];
  int c1i[128], d2i[NP * 128];

  // fp8, two variants: (v0) A1 = 0, C1 = NaN-free random f32 bits, so D1 = C1 + 0; (v1) C1 = 0, A1/B1 small integers.
  for (int v = 0; v < 2; v++) {
    for (int i = 0; i < 128; i++) {
      if (v == 0) {
        uint32_t w;
        do {
          w = (uint32_t)rng();
          uint32_t ex = (w >> 23) & 0xff;
          bool bad = ex == 0 || ex >= 254;
          for (int y = 0; y < 4; y++) { uint8_t by = w >> (8 * y); bad |= (by & 0x7f) == 0x7f; }
          if (!bad) break;
        } while (true);
        memcpy(&c1f[i], &w, 4);
        a1[i] = 0;
      } else {
        c1f[i] = 0.f;
        uint32_t w = 0;
        for (int y = 0; y < 4; y++) w |= (uint32_t)e4m3_of((double)((int)(rng() % 9) - 4)) << (8 * y);
        a1[i] = w;
      }
    }
    for (int i = 0; i < 64; i++) {
      uint32_t w = 0;
      for (int y = 0; y < 4; y++) w |= (uint32_t)e4m3_of((double)((int)(rng() % 9) - 4)) << (8 * y);
      b1[i] = v == 0 ? 0 : w;
    }
    sel_b(b2, NP, 0x38);
    CK(cudaMemcpy(da1, a1, sizeof a1, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db1, b1, sizeof b1, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc1f, c1f, sizeof c1f, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db2, b2, sizeof b2, cudaMemcpyHostToDevice));
    k_fp8<<<1, 32>>>(da1, db1, dc1f, db2, NP, dd1, dd2f);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(d1, dd1, sizeof d1, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(d2f, dd2f, sizeof d2f, cudaMemcpyDeviceToHost));
    long checked = 0, mism = 0, nanrows = 0, d1bad = 0;
    if (v == 0) for (int i = 0; i < 128; i++) { uint32_t w; memcpy(&w, &c1f[i], 4); d1bad += d1[i] != w; }
    for (int p = 0; p < NP; p++)
      for (int l = 0; l < 32; l++)
        for (int r = 0; r < 4; r++) {
          int g = l / 4, t = l % 4, i = g + 8 * (r >= 2), j = 2 * t + (r & 1);
          bool rownan = false;
          for (int k = 0; k < 32; k++) rownan |= std::isnan(e4m3(a_elem(d1, i, k)));
          if (rownan) { nanrows++; continue; }
          double want = e4m3(a_elem(d1, i, sel(p, j)));
          checked++;
          mism += (double)d2f[(p * 32 + l) * 4 + r] != want;
        }
    printf("{\"probe\":\"reinterp-fp8\",\"variant\":%d,\"elements_checked\":%ld,\"mismatches\":%ld,"
           "\"skipped_nan_rows\":%ld,\"d1_passthrough_mismatches\":%ld}\n", v, checked, mism, nanrows, d1bad);
  }

  // s8: A1 = 0, C1 = v uniform in [-1000, 1000], so D1 = v; D2 reads D1's bytes as s8.
  for (int i = 0; i < 128; i++) { c1i[i] = (int)(rng() % 2001) - 1000; a1[i] = 0; }
  for (int i = 0; i < 64; i++) b1[i] = 0;
  sel_b(b2, NP, 0x01);
  CK(cudaMemcpy(da1, a1, sizeof a1, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db1, b1, sizeof b1, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc1i, c1i, sizeof c1i, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db2, b2, sizeof b2, cudaMemcpyHostToDevice));
  k_s8<<<1, 32>>>(da1, db1, dc1i, db2, NP, dd1, dd2i);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(d1, dd1, sizeof d1, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(d2i, dd2i, sizeof d2i, cudaMemcpyDeviceToHost));
  long checked = 0, mism = 0, copies = 0, copies_ok = 0, d1bad = 0;
  for (int i = 0; i < 128; i++) d1bad += (int)d1[i] != c1i[i];
  for (int p = 0; p < NP; p++)
    for (int l = 0; l < 32; l++)
      for (int r = 0; r < 4; r++) {
        int g = l / 4, t = l % 4, i = g + 8 * (r >= 2), j = 2 * t + (r & 1), k = sel(p, j);
        int want = (int)(int8_t)a_elem(d1, i, k);
        int got = d2i[(p * 32 + l) * 4 + r];
        checked++;
        mism += got != want;
        if (k % 4 == 0) {
          int lane = (i % 8) * 4 + (k % 16) / 4, rr = (i >= 8) + 2 * (k >= 16);
          int src = (int)d1[lane * 4 + rr];
          if (src >= -128 && src <= 127) { copies++; copies_ok += got == src; }
        }
      }
  printf("{\"probe\":\"reinterp-s8\",\"elements_checked\":%ld,\"mismatches\":%ld,\"exact_copies_tested\":%ld,"
         "\"exact_copies_ok\":%ld,\"d1_passthrough_mismatches\":%ld}\n", checked, mism, copies, copies_ok, d1bad);
  return 0;
}
