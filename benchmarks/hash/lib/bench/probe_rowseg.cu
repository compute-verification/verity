// Temporary probe: k_rows_seg and variants against k_row_segs + k_folds (the two launches) on this GPU, for RTX PRO 6000's
// wrong row-seg digests at an even segment count (r20261018-225819-7f1f, r20261018-231135-6d8d). Removed once it has answered.
#include "../vc.cu"

#include <cstdio>
#include <random>

namespace {

struct Launch {
  u32 s, threads, g;
  unsigned grid;
  size_t smem;
};
Launch shape(u32 s, u64 n) {
  const u32 threads = s < 128 ? (s + 31) / 32 * 32 : 128, g = s < threads ? threads / s : 1;
  return {s, threads, g, blocks(n, g), 64ull * g * s};
}

// D: k_rows_seg, with every segment digest copied out of shared memory into dbg (k_row_segs' layout)
__global__ void __launch_bounds__(128) k_dbg(const __grid_constant__ RowK R, const uint8_t* __restrict__ tab,
                                             const uint8_t* __restrict__ rows, u64 stride, u64 n, uint8_t* __restrict__ out,
                                             u64* dbg) {
  extern __shared__ u64 sdig[];
  const u32 s = R.s[0], g = s < blockDim.x ? blockDim.x / s : 1;
  const u64 r0 = (u64)blockIdx.x * g;
  for (u32 t = threadIdx.x; t < g * s; t += blockDim.x) {
    const u32 q = t / s, j = t - q * s;
    if (r0 + q < n) seg_hash(R, tab, 0, j, rows + (r0 + q) * stride + (u64)SEG * j, sdig + 8 * t);
  }
  __syncthreads();
  for (u32 t = threadIdx.x; t < g * s; t += blockDim.x)
    if (r0 + t / s < n)
      for (int k = 0; k < 8; ++k) dbg[8 * (r0 * s + t) + k] = sdig[8 * t + k];
  if (threadIdx.x < g && r0 + threadIdx.x < n) seg_fold<true>(R.mid_Q[0], sdig + 8ull * s * threadIdx.x, s, out + 64 * (r0 + threadIdx.x));
}

// S: static shared memory (s <= 128)
__global__ void __launch_bounds__(128) k_static(const __grid_constant__ RowK R, const uint8_t* __restrict__ tab,
                                                const uint8_t* __restrict__ rows, u64 stride, u64 n, uint8_t* __restrict__ out) {
  __shared__ u64 sdig[128 * 8];
  const u32 s = R.s[0], g = s < blockDim.x ? blockDim.x / s : 1;
  const u64 r0 = (u64)blockIdx.x * g;
  for (u32 t = threadIdx.x; t < g * s; t += blockDim.x) {
    const u32 q = t / s, j = t - q * s;
    if (r0 + q < n) seg_hash(R, tab, 0, j, rows + (r0 + q) * stride + (u64)SEG * j, sdig + 8 * t);
  }
  __syncthreads();
  if (threadIdx.x < g && r0 + threadIdx.x < n) seg_fold<true>(R.mid_Q[0], sdig + 8ull * s * threadIdx.x, s, out + 64 * (r0 + threadIdx.x));
}

// G: the digests through global memory (dg, plain loads after the barrier), no shared memory
__global__ void __launch_bounds__(128) k_glob(const __grid_constant__ RowK R, const uint8_t* __restrict__ tab,
                                              const uint8_t* __restrict__ rows, u64 stride, u64 n, uint8_t* __restrict__ out, u64* dg) {
  const u32 s = R.s[0], g = s < blockDim.x ? blockDim.x / s : 1;
  const u64 r0 = (u64)blockIdx.x * g;
  for (u32 t = threadIdx.x; t < g * s; t += blockDim.x) {
    const u32 q = t / s, j = t - q * s;
    if (r0 + q < n) seg_hash(R, tab, 0, j, rows + (r0 + q) * stride + (u64)SEG * j, dg + 8 * (r0 * s + t));
  }
  __syncthreads();
  if (threadIdx.x < g && r0 + threadIdx.x < n)
    seg_fold<true>(R.mid_Q[0], dg + 8 * (r0 * s + (u64)s * threadIdx.x), s, out + 64 * (r0 + threadIdx.x));
}

// Q: k_rows_seg with the Q state in shared memory (as s512_rows keeps it), not read through the parameter's address
__global__ void __launch_bounds__(128) k_qsh(const __grid_constant__ RowK R, const uint8_t* __restrict__ tab,
                                             const uint8_t* __restrict__ rows, u64 stride, u64 n, uint8_t* __restrict__ out) {
  extern __shared__ u64 sdig[];
  __shared__ u64 qst[8];
  const u32 s = R.s[0], g = s < blockDim.x ? blockDim.x / s : 1;
  const u64 r0 = (u64)blockIdx.x * g;
  if (threadIdx.x < 8) qst[threadIdx.x] = R.mid_Q[0][threadIdx.x];
  for (u32 t = threadIdx.x; t < g * s; t += blockDim.x) {
    const u32 q = t / s, j = t - q * s;
    if (r0 + q < n) seg_hash(R, tab, 0, j, rows + (r0 + q) * stride + (u64)SEG * j, sdig + 8 * t);
  }
  __syncthreads();
  if (threadIdx.x < g && r0 + threadIdx.x < n) seg_fold<true>(qst, sdig + 8ull * s * threadIdx.x, s, out + 64 * (r0 + threadIdx.x));
}

// K: k_rows_seg with an even s's last block, all padding, through compress_kw from the host's schedule (a kernel parameter)
struct KwP {
  u64 kw[80];
};
__global__ void __launch_bounds__(128) k_kw(const __grid_constant__ RowK R, const __grid_constant__ KwP P, const uint8_t* __restrict__ tab,
                                            const uint8_t* __restrict__ rows, u64 stride, u64 n, uint8_t* __restrict__ out) {
  extern __shared__ u64 sdig[];
  const u32 s = R.s[0], g = s < blockDim.x ? blockDim.x / s : 1;
  const u64 r0 = (u64)blockIdx.x * g;
  for (u32 t = threadIdx.x; t < g * s; t += blockDim.x) {
    const u32 q = t / s, j = t - q * s;
    if (r0 + q < n) seg_hash(R, tab, 0, j, rows + (r0 + q) * stride + (u64)SEG * j, sdig + 8 * t);
  }
  __syncthreads();
  if (threadIdx.x >= g || r0 + threadIdx.x >= n) return;
  const u64* dig = sdig + 8ull * s * threadIdx.x;
  u64 st[8], w[16];
  cpy8(st, R.mid_Q[0]);
  const int pairs = (int)(s / 2);
#pragma unroll 1
  for (int k = 0; k < pairs; ++k) {
    ld_dig<true>(dig + 16 * k, w, 16);
    sha::compress(st, w);
  }
  if (s & 1) {
    ld_dig<true>(dig + 16 * pairs, w, 8);
#pragma unroll
    for (int t = 8; t < 16; ++t) w[t] = 0;
    w[8] = 0x8000000000000000ull;
    w[15] = 8ull * (128 + 64ull * s);
    sha::compress(st, w);
  } else {
    sha::compress_kw(st, P.kw);
  }
  st_be(out + 64 * (r0 + threadIdx.x), st, 8);
}

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_) { printf("CUDA %d %s at %d\n", (int)e_, cudaGetErrorString(e_), __LINE__); exit(1); } \
  } while (0)

std::vector<uint8_t> down(const void* d, size_t n) {
  std::vector<uint8_t> h(n);
  CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost));
  return h;
}
int rows_wrong(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, u64 n, char* mask) {
  int w = 0;
  for (u64 i = 0; i < n; ++i) {
    const bool bad = memcmp(a.data() + 64 * i, b.data() + 64 * i, 64) != 0;
    w += bad;
    if (i < 63) mask[i] = bad ? 'x' : '.', mask[i + 1] = 0;
  }
  return w;
}

}  // namespace

int main() {
  CK((cudaError_t)vc_init());
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("device %s sm_%d%d SHA_ADD %d\n", prop.name, prop.major, prop.minor, SHA_ADD);
  std::mt19937_64 rng(7);
  for (u32 s : {1u, 2u, 3u, 4u, 8u, 16u, 31u, 32u, 56u, 57u, 64u, 128u, 201u, 202u}) {
    const u64 n = 9, len = 1024ull * s;
    std::vector<uint8_t> rows(n * len), did(64), pre(128);
    for (auto& b : rows) b = (uint8_t)rng();
    for (auto& b : did) b = (uint8_t)rng();
    for (auto& b : pre) b = (uint8_t)rng();
    vc_spec sp = {};
    sp.domain_id = did.data();
    sp.row_scheme = VC_ROW_SEG;
    sp.row_bits = 8 * len;
    sp.row_prefix = pre.data();
    std::vector<uint8_t> plan(vc_plan_bytes());
    if (vc_plan_build(&sp, plan.data())) { printf("s %u: plan refused\n", s); continue; }
    const Plan* p = plan_of(plan.data());
    uint8_t *tab, *dr, *o_ref, *o_a, *o_d, *o_s, *o_g, *o_q, *o_k;
    u64 *dig_ref, *dbg, *dg;
    CK(cudaMalloc(&tab, vc_tables_bytes(plan.data())));
    CK((cudaError_t)vc_tables_launch(plan.data(), tab, 0));
    CK(cudaMalloc(&dr, rows.size()));
    CK(cudaMemcpy(dr, rows.data(), rows.size(), cudaMemcpyHostToDevice));
    for (uint8_t** o : {&o_ref, &o_a, &o_d, &o_s, &o_g, &o_q, &o_k}) { CK(cudaMalloc(o, 64 * n)); CK(cudaMemset(*o, 0xA5, 64 * n)); }
    for (u64** d : {&dig_ref, &dbg, &dg}) { CK(cudaMalloc(d, 64 * n * s)); CK(cudaMemset(*d, 0xA5, 64 * n * s)); }
    k_row_segs<<<blocks(n * s, 128), 128>>>(p->row, tab, dr, len, n, dig_ref);
    k_folds<<<blocks(n, 32), 32>>>(p->row, dig_ref, n, n, o_ref);
    CK(cudaGetLastError());
    CK((cudaError_t)vc_rows_launch(plan.data(), tab, dr, len, n, o_a, dbg, 0));
    const Launch L = shape(s, n);
    k_dbg<<<L.grid, L.threads, L.smem>>>(p->row, tab, dr, len, n, o_d, dbg);
    if (s <= 128) k_static<<<L.grid, L.threads>>>(p->row, tab, dr, len, n, o_s);
    k_glob<<<L.grid, L.threads>>>(p->row, tab, dr, len, n, o_g, dg);
    k_qsh<<<L.grid, L.threads, L.smem>>>(p->row, tab, dr, len, n, o_q);
    KwP P;
    pad_schedule(128 + 64ull * s, P.kw);
    k_kw<<<L.grid, L.threads, L.smem>>>(p->row, P, tab, dr, len, n, o_k);
    CK(cudaDeviceSynchronize());
    const auto ref = down(o_ref, 64 * n);
    char m[6][64];
    const int wa = rows_wrong(down(o_a, 64 * n), ref, n, m[0]), wd = rows_wrong(down(o_d, 64 * n), ref, n, m[1]),
              ws = s <= 128 ? rows_wrong(down(o_s, 64 * n), ref, n, m[2]) : -1, wg = rows_wrong(down(o_g, 64 * n), ref, n, m[3]),
              wq = rows_wrong(down(o_q, 64 * n), ref, n, m[4]), wk = rows_wrong(down(o_k, 64 * n), ref, n, m[5]);
    const auto dref = down(dig_ref, 64 * n * s), ddbg = down(dbg, 64 * n * s), dgg = down(dg, 64 * n * s);
    int segs_bad = 0, segs_bad_g = 0;
    for (u64 t = 0; t < n * s; ++t) {
      segs_bad += memcmp(dref.data() + 64 * t, ddbg.data() + 64 * t, 64) != 0;
      segs_bad_g += memcmp(dref.data() + 64 * t, dgg.data() + 64 * t, 64) != 0;
    }
    const auto a = down(o_a, 64 * n);
    bool poison = true;
    for (int k = 0; k < 64; ++k) poison &= a[k] == 0xA5;
    printf("s %3u threads %3u g %3u | rows wrong: lib %d [%s] dbg %d static %d glob %d qsh %d kw %d | shared segs wrong %d, glob segs "
           "wrong %d | lib row 0 %s\n",
           s, L.threads, L.g, wa, m[0], wd, ws, wg, wq, wk, segs_bad, segs_bad_g, poison ? "unwritten" : "written");
    for (void* d : {(void*)tab, (void*)dr, (void*)o_ref, (void*)o_a, (void*)o_d, (void*)o_s, (void*)o_g, (void*)o_q, (void*)o_k, (void*)dig_ref,
                    (void*)dbg, (void*)dg})
      CK(cudaFree(d));
  }
  return 0;
}
