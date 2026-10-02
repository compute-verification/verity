// The numeric pin of the SASS gate (sass_gate.py; sass_pins.json "numeric"). ptxas builds div.rn, sqrt.rn and rcp.rn from
// .FTZ instructions and MUFU seeds, and integer division from a MUFU.RCP seed, in every build; the gate exempts them only as a
// pinned ptxas builds them, and this is the evidence that each is exact there, subnormals included; its builds are where
// sass_pins.json's pins come from. Each check runs a sequence as this file compiles it (ieee_pin.py gates the binary and
// requires every sequence and every body its toolkit pins) and compares it on the card with a reference that doesn't use it:
//   rcp    __frcp_rn(x) on all 2^32 x, against 1 / x in double rounded once (innocuous double rounding: 53 >= 2*24 + 2)
//   sqrt   sqrtf(x) on all 2^32 x, against sqrt in double rounded once
//   div    a / b on 2^32 counter-hashed pairs weighted to subnormal operands and results, short mantissas (exact quotients and
//          ties) and the overflow edge, against a / b in double rounded once
//   cdiv   3.0f / b on all 2^32 b (cdiv3) beside a / b on weighted a (cdiv), in one kernel, so the two share their slow path
//          and the constant is an immediate of the fast path, as ptxas builds it outside a function with one division
//   u16    div.u16 on all 2^32 pairs in a loop, and again (u16_call) one pair per thread from memory, the two call sites for
//          which 13.0 builds __cuda_sm20_div_u16 differently; u32 a / b, s32 a / b and a % b, u32n a / n for a uniform n (64
//          divisors), u64 a / b and s64 a % b on weighted pairs: exact when q * b + r == a with |r| < |b| and r of a's sign,
//          in integer arithmetic
//   host_* the device's rcp, sqrt, div and 3 / b on every subnormal input and every input whose reciprocal is subnormal
//          (2^24 - 2 + 2^25), each with a weighted divisor, against the host's double arithmetic: a reference off the card
// Division by zero and INT_MIN / -1 are outside the domain, counted as skipped. NaN results compare equal whatever their
// payload; everything else bit for bit. Prints one JSON line for the device, then one per check.
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a --no-compress [FLAGS] ieee_pin.cu -o ieee_pin
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define EACH(i, n) for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (uint64_t)gridDim.x * blockDim.x)

struct Tally {
  unsigned long long cases, skipped, sub_in, sub_out, bad, first[4];
  int got;
};

__host__ __device__ inline uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
__host__ __device__ inline float flt(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
__host__ __device__ inline bool same(float h, float r) { return bits(h) == bits(r) || (h != h && r != r); }
__host__ __device__ inline bool subn(float x) { uint32_t u = bits(x); return (u & 0x7F800000u) == 0 && (u & 0x7FFFFFu); }

__host__ __device__ inline uint32_t mix(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
  return (uint32_t)x;
}

// An FP32 operand: 1/8 subnormal, 1/8 just above, 1/8 near overflow, 1/8 around 1, the rest any exponent; 1/8 of each with a
// short mantissa (exact quotients and ties).
__host__ __device__ inline uint32_t pick(uint64_t i, uint32_t salt) {
  uint32_t r = mix(i * 2 + salt), cls = r & 7, man = mix(i * 7 + salt + 99) & 0x7FFFFF, e;
  if (cls == 0) e = 0;
  else if (cls == 1) e = 1 + (r >> 8) % 8;
  else if (cls == 2) e = 247 + (r >> 8) % 8;
  else if (cls == 3) e = 120 + (r >> 8) % 16;
  else e = 1 + (r >> 8) % 254;
  if (((r >> 16) & 7) == 0) man &= 0x7F0000;
  return (r & 0x80000000u) | (e << 23) | man;
}

// Integer operands: small, 16-bit, powers of two and their neighbours, the top of the range, a shifted value, or any.
__host__ __device__ inline uint32_t pick32(uint64_t i, uint32_t salt) {
  uint32_t r = mix(i * 2 + salt), v = mix(i * 7 + salt + 99), k = (r >> 8) & 31;
  switch (r & 7) {
    case 0: return v & 0xff;
    case 1: return v & 0xffff;
    case 2: return 1u << k;
    case 3: return (1u << k) - 1u;
    case 4: return (1u << k) + 1u;
    case 5: return ~(v & 0xff);
    case 6: return v >> k;
    default: return v;
  }
}

// As pick32 at 64 bits; 1/8 fit in 32 bits, the path ptxas inlines instead of calling __cuda_sm20_div_u64.
__host__ __device__ inline uint64_t pick64(uint64_t i, uint32_t salt) {
  uint32_t r = mix(i * 2 + salt), k = (r >> 8) & 63;
  uint64_t v = (uint64_t)mix(i * 7 + salt + 99) << 32 | mix(i * 11 + salt + 7);
  switch (r & 7) {
    case 0: return v & 0xffffffffull;
    case 1: return v & 0xffff;
    case 2: return 1ull << k;
    case 3: return (1ull << k) - 1;
    case 4: return (1ull << k) + 1;
    case 5: return ~(v & 0xff);
    case 6: return v >> k;
    default: return v;
  }
}

__device__ void add(Tally* t, unsigned long long c, unsigned long long s, unsigned long long si, unsigned long long so,
                    unsigned long long b) {
  atomicAdd(&t->cases, c); atomicAdd(&t->skipped, s); atomicAdd(&t->sub_in, si); atomicAdd(&t->sub_out, so);
  atomicAdd(&t->bad, b);
}

__device__ void first(Tally* t, uint64_t a, uint64_t b, uint64_t h, uint64_t r) {
  if (atomicCAS(&t->got, 0, 1) == 0) { t->first[0] = a; t->first[1] = b; t->first[2] = h; t->first[3] = r; }
}

__global__ void rcp_all(Tally* t) {
  unsigned long long c = 0, si = 0, so = 0, nb = 0;
  EACH(i, 1ull << 32) {
    float x = flt((uint32_t)i), h = __frcp_rn(x), r = __double2float_rn(1.0 / (double)x);
    c++; si += subn(x); so += subn(r);
    if (!same(h, r)) { nb++; first(t, bits(x), 0, bits(h), bits(r)); }
  }
  add(t, c, 0, si, so, nb);
}

__global__ void sqrt_all(Tally* t) {
  unsigned long long c = 0, si = 0, so = 0, nb = 0;
  EACH(i, 1ull << 32) {
    float x = flt((uint32_t)i), h = sqrtf(x), r = __double2float_rn(sqrt((double)x));
    c++; si += subn(x); so += subn(r);
    if (!same(h, r)) { nb++; first(t, bits(x), 0, bits(h), bits(r)); }
  }
  add(t, c, 0, si, so, nb);
}

__global__ void div_all(Tally* t) {
  unsigned long long c = 0, si = 0, so = 0, nb = 0;
  EACH(i, 1ull << 32) {
    float a = flt(pick(i, 1)), b = flt(pick(i, 2)), h = a / b, r = __double2float_rn((double)a / (double)b);
    c++; si += subn(a) || subn(b); so += subn(r);
    if (!same(h, r)) { nb++; first(t, bits(a), bits(b), bits(h), bits(r)); }
  }
  add(t, c, 0, si, so, nb);
}

__global__ void cdiv_all(Tally* t) {
  unsigned long long c = 0, si[2] = {0, 0}, so[2] = {0, 0}, nb[2] = {0, 0};
  EACH(i, 1ull << 32) {
    float b = flt((uint32_t)i), a = flt(pick(i, 3)), h3 = 3.0f / b, ha = a / b;
    float r3 = __double2float_rn(3.0 / (double)b), ra = __double2float_rn((double)a / (double)b);
    c++; si[0] += subn(b); so[0] += subn(r3); si[1] += subn(a) || subn(b); so[1] += subn(ra);
    if (!same(h3, r3)) { nb[0]++; first(t, bits(b), 0, bits(h3), bits(r3)); }
    if (!same(ha, ra)) { nb[1]++; first(t + 1, bits(a), bits(b), bits(ha), bits(ra)); }
  }
  add(t, c, 0, si[0], so[0], nb[0]);
  add(t + 1, c, 0, si[1], so[1], nb[1]);
}

__global__ void u16_all(Tally* t) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, 1ull << 32) {
    uint16_t a = (uint16_t)i, b = (uint16_t)(i >> 16), q;
    if (b == 0) { s++; continue; }
    asm("div.u16 %0, %1, %2;" : "=h"(q) : "h"(a), "h"(b));
    uint32_t p = (uint32_t)q * b;
    c++;
    if (!(p <= a && a - p < b)) { nb++; first(t, a, b, q, 0); }
  }
  add(t, c, s, 0, 0, nb);
}

// ptxas specializes __cuda_sm20_div_u16 to its call sites: with the operands loaded straight into the call, as in
// ftz_probe.cu's udiv16_k, 13.0 converts the dividend as a u32 (I2FP.F32.U32.RZ) instead of as a u16 (I2F.U16.RZ, as in the
// loop above). u16_call divides N pairs that u16_fill writes from `base`, and u16_check tallies them.
__global__ void u16_fill(uint16_t* a, uint16_t* b, uint64_t base, uint64_t n) {
  EACH(j, n) { a[j] = (uint16_t)(base + j); b[j] = (uint16_t)((base + j) >> 16); }
}

__global__ void u16_call(uint16_t* y, const uint16_t* a, const uint16_t* b) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  uint16_t r;
  asm("div.u16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  y[i] = r;
}

__global__ void u16_check(Tally* t, const uint16_t* y, const uint16_t* a, const uint16_t* b, uint64_t n) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(j, n) {
    if (b[j] == 0) { s++; continue; }
    uint32_t p = (uint32_t)y[j] * b[j];
    c++;
    if (!(p <= a[j] && a[j] - p < b[j])) { nb++; first(t, a[j], b[j], y[j], 0); }
  }
  add(t, c, s, 0, 0, nb);
}

__global__ void u32_all(Tally* t) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, 1ull << 32) {
    uint32_t a = pick32(i, 1), b = pick32(i, 2), r2 = mix(i * 5 + 3);
    if ((r2 & 7) == 0) b = (a >> ((r2 >> 3) & 31)) + ((r2 >> 8) & 3) - 1;
    if (b == 0) { s++; continue; }
    uint32_t q = a / b;
    uint64_t p = (uint64_t)q * b;
    c++;
    if (!(p <= a && a - p < b)) { nb++; first(t, a, b, q, 0); }
  }
  add(t, c, s, 0, 0, nb);
}

__global__ void s32_all(Tally* t) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, 1ull << 32) {
    uint32_t r2 = mix(i * 5 + 4);
    int32_t a = (int32_t)pick32(i, 3), b = (int32_t)pick32(i, 4);
    if (r2 & 1) a = (int32_t)(0u - (uint32_t)a);
    if (r2 & 2) b = (int32_t)(0u - (uint32_t)b);
    if (b == 0 || (a == INT32_MIN && b == -1)) { s++; continue; }
    int32_t q = a / b, r = a % b;
    int64_t ar = r < 0 ? -(int64_t)r : r, ab = b < 0 ? -(int64_t)b : b;
    c++;
    if (!((int64_t)q * b + r == a && ar < ab && (r == 0 || (r < 0) == (a < 0)))) {
      nb++; first(t, (uint32_t)a, (uint32_t)b, (uint32_t)q, (uint32_t)r);
    }
  }
  add(t, c, s, 0, 0, nb);
}

__global__ void u32n_all(Tally* t, uint32_t n, uint64_t count, uint32_t salt) {
  unsigned long long c = 0, nb = 0;
  EACH(i, count) {
    uint32_t a = pick32(i, salt), q = a / n;
    uint64_t p = (uint64_t)q * n;
    c++;
    if (!(p <= a && a - p < n)) { nb++; first(t, a, n, q, 0); }
  }
  add(t, c, 0, 0, 0, nb);
}

__global__ void u64_all(Tally* t, uint64_t count) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, count) {
    uint64_t a = pick64(i, 5), b = pick64(i, 6);
    if (b == 0) { s++; continue; }
    uint64_t q = a / b, lo = q * b, hi = __umul64hi(q, b);
    c++;
    if (!(hi == 0 && lo <= a && a - lo < b)) { nb++; first(t, a, b, q, 0); }
  }
  add(t, c, s, 0, 0, nb);
}

__global__ void s64_all(Tally* t, uint64_t count) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, count) {
    uint32_t r2 = mix(i * 13 + 1);
    int64_t a = (int64_t)pick64(i, 7), b = (int64_t)pick64(i, 8);
    if (r2 & 1) a = (int64_t)(0 - (uint64_t)a);
    if (r2 & 2) b = (int64_t)(0 - (uint64_t)b);
    if (b == 0 || (a == INT64_MIN && b == -1)) { s++; continue; }
    int64_t r = a % b;
    uint64_t ua = a < 0 ? 0 - (uint64_t)a : (uint64_t)a, ub = b < 0 ? 0 - (uint64_t)b : (uint64_t)b;
    uint64_t uq = ua / ub, lo = uq * ub, hi = __umul64hi(uq, ub), ur = ua - lo;
    int64_t want = a < 0 ? (int64_t)(0 - ur) : (int64_t)ur;
    c++;
    if (!(hi == 0 && lo <= ua && ur < ub && r == want)) { nb++; first(t, (uint64_t)a, (uint64_t)b, (uint64_t)r, (uint64_t)want); }
  }
  add(t, c, s, 0, 0, nb);
}

__global__ void apply(const uint32_t* x, const uint32_t* y, uint32_t* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = flt(x[i]), b = flt(y[i]);
    out[4 * i] = bits(__frcp_rn(a)); out[4 * i + 1] = bits(sqrtf(a)); out[4 * i + 2] = bits(a / b);
    out[4 * i + 3] = bits(3.0f / b);
  }
}

static void print(const char* check, const Tally& t) {
  printf("{\"check\":\"%s\",\"cases\":%llu,\"skipped\":%llu,\"subnormal_in\":%llu,\"subnormal_out\":%llu,\"bad\":%llu,"
         "\"first\":[%llu,%llu,%llu,%llu]}\n", check, t.cases, t.skipped, t.sub_in, t.sub_out, t.bad, t.first[0], t.first[1],
         t.first[2], t.first[3]);
  fflush(stdout);
}

static void host_check() {
  std::vector<uint32_t> x, y;
  for (uint32_t s = 0; s < 2; s++)
    for (uint32_t m = 1; m < (1u << 23); m++) x.push_back(s << 31 | m);
  for (uint32_t s = 0; s < 2; s++)
    for (uint32_t e = 253; e <= 254; e++)
      for (uint32_t m = 0; m < (1u << 23); m++) x.push_back(s << 31 | e << 23 | m);
  for (size_t i = 0; i < x.size(); i++) y.push_back(pick(i, 2));
  size_t n = x.size();
  uint32_t *dx, *dy, *dout;
  CK(cudaMalloc(&dx, n * 4)); CK(cudaMalloc(&dy, n * 4)); CK(cudaMalloc(&dout, n * 16));
  CK(cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dy, y.data(), n * 4, cudaMemcpyHostToDevice));
  apply<<<(unsigned)((n + 255) / 256), 256>>>(dx, dy, dout, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<uint32_t> out(n * 4);
  CK(cudaMemcpy(out.data(), dout, n * 16, cudaMemcpyDeviceToHost));
  CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(dout));
  Tally t[4] = {};
  for (size_t i = 0; i < n; i++) {
    float a = flt(x[i]), b = flt(y[i]);
    float ref[4] = {(float)(1.0 / (double)a), (float)std::sqrt((double)a), (float)((double)a / (double)b),
                    (float)(3.0 / (double)b)};
    bool in[4] = {subn(a), subn(a), subn(a) || subn(b), subn(b)};
    for (int k = 0; k < 4; k++) {
      float h = flt(out[4 * i + k]);
      t[k].cases++; t[k].sub_in += in[k]; t[k].sub_out += subn(ref[k]);
      if (!same(h, ref[k])) {
        if (!t[k].bad) { t[k].first[0] = x[i]; t[k].first[1] = y[i]; t[k].first[2] = bits(h); t[k].first[3] = bits(ref[k]); }
        t[k].bad++;
      }
    }
  }
  print("host_rcp", t[0]); print("host_sqrt", t[1]); print("host_div", t[2]); print("host_cdiv3", t[3]);
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const unsigned char* u = (const unsigned char*)p.uuid.bytes;
  printf("{\"device\":\"%s\",\"uuid\":\"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\","
         "\"sm\":%d%d,\"sms\":%d}\n", p.name, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12],
         u[13], u[14], u[15], p.major, p.minor, p.multiProcessorCount);
  const int grid = p.multiProcessorCount * 8;
  Tally* t;
  CK(cudaMallocManaged(&t, 2 * sizeof(Tally)));
  auto run = [&](const char* check, auto launch) {
    memset(t, 0, 2 * sizeof(Tally));
    launch();
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    print(check, t[0]);
  };
  run("rcp", [&] { rcp_all<<<grid, 256>>>(t); });
  run("sqrt", [&] { sqrt_all<<<grid, 256>>>(t); });
  run("div", [&] { div_all<<<grid, 256>>>(t); });
  memset(t, 0, 2 * sizeof(Tally));
  cdiv_all<<<grid, 256>>>(t);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  print("cdiv3", t[0]); print("cdiv", t[1]);
  run("u16", [&] { u16_all<<<grid, 256>>>(t); });
  run("u16_call", [&] {
    const uint64_t n = 1ull << 28;
    uint16_t *a, *b, *y;
    CK(cudaMalloc(&a, n * 2)); CK(cudaMalloc(&b, n * 2)); CK(cudaMalloc(&y, n * 2));
    for (uint64_t base = 0; base < (1ull << 32); base += n) {
      u16_fill<<<grid, 256>>>(a, b, base, n);
      u16_call<<<(unsigned)(n / 256), 256>>>(y, a, b);
      u16_check<<<grid, 256>>>(t, y, a, b, n);
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(y));
  });
  run("u32", [&] { u32_all<<<grid, 256>>>(t); });
  run("s32", [&] { s32_all<<<grid, 256>>>(t); });
  run("u32n", [&] {
    const uint32_t fixed[] = {1u, 2u, 3u, 5u, 7u, 10u, 255u, 256u, 257u, 0xffffu, 0x10000u, 0x10001u, 0x7fffffffu,
                              0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu};
    for (uint32_t k = 0; k < 64; k++) {
      uint32_t n = k < 17 ? fixed[k] : (mix(k) >> (mix(k + 1000) & 31)) | 1u;
      u32n_all<<<grid, 256>>>(t, n, 1ull << 26, k + 1);
    }
  });
  run("u64", [&] { u64_all<<<grid, 256>>>(t, 1ull << 30); });
  run("s64", [&] { s64_all<<<grid, 256>>>(t, 1ull << 30); });
  host_check();
  return 0;
}
