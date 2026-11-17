// `ncp2.cuh` on the device (`pouw_device.device`): the committer's first, correctness-first form, and the hashing throughput
// kernels the GPU-path measurement times. Every entry takes host pointers and copies, so `pouw_device` drives it through
// ctypes exactly as it drives the host build, and the gate is the same code. The committer keeps route U's checkpoints in
// global memory and hashes each unit from there; the serving kernel fuses them (docs/pouw/gpu-path-plan.md).
#include <cuda_runtime.h>

#include "ncp2.cuh"

using namespace ncp2;

#define CK(x)                            \
  do {                                   \
    cudaError_t e_ = (x);                \
    if (e_ != cudaSuccess) return (int)e_; \
  } while (0)

template <class T>
struct Dev {
  T* p = nullptr;
  ~Dev() { if (p) cudaFree(p); }
  cudaError_t alloc(size_t n) { return cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T)); }
  cudaError_t put(const T* h, size_t n) {
    cudaError_t e = alloc(n);
    return e != cudaSuccess ? e : cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice);
  }
};

static unsigned blocks(size_t n, unsigned per) { return (unsigned)((n + per - 1) / per); }

// ---- hashing -------------------------------------------------------------------------------------------------------------

__global__ void k_hash(int kind, int ds, const uint8_t* data, const uint64_t* off, uint64_t n, uint64_t outlen, uint8_t* out,
                       int* rc) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  int r = hash_one(kind, ds, data + off[i], (size_t)(off[i + 1] - off[i]), (size_t)outlen, out + i * outlen);
  if (r) atomicExch(rc, r);
}

// `ncp2_host.cpp`'s `ncp2_hash` on the device, one thread per message.
extern "C" int ncp2_hash(int kind, int ds, const uint8_t* data, const uint64_t* offsets, uint64_t n, uint64_t outlen, uint8_t* out) {
  Dev<uint8_t> d, o;
  Dev<uint64_t> f;
  Dev<int> rc;
  int h = 0;
  CK(d.put(data, (size_t)offsets[n]));
  CK(f.put(offsets, (size_t)n + 1));
  CK(o.alloc((size_t)(n * outlen)));
  CK(rc.put(&h, 1));
  k_hash<<<blocks(n, 128), 128>>>(kind, ds, d.p, f.p, n, outlen, o.p, rc.p);
  CK(cudaGetLastError());
  CK(cudaMemcpy(out, o.p, (size_t)(n * outlen), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&h, rc.p, sizeof(int), cudaMemcpyDeviceToHost));
  return h;
}

// ---- the committer --------------------------------------------------------------------------------------------------------

__global__ void k_row_digests(const int8_t* a, int m, int k, uint8_t* rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) row_digest(a + (size_t)i * k, k, rows + 64 * (size_t)i);
}

__global__ void k_strips(const uint8_t* rows, int ms, uint8_t* strips, uint8_t* leaves) {
  int st = blockIdx.x * blockDim.x + threadIdx.x;
  if (st >= ms) return;
  strip_digest(rows + (size_t)st * D * 64, strips + 64 * (size_t)st);
  unit_leaf_digest(strips + 64 * (size_t)st, leaves + 32 * (size_t)st);
}

// One thread: the tree, its node units, the key and its unit.
__global__ void k_tree_key(const uint8_t* strips, int ms, uint8_t* nodes, const uint8_t* salt, uint64_t forward,
                           uint64_t position, uint64_t weight, uint8_t* key, uint8_t* leaves) {
  int count = 0;
  const uint8_t* da = rfc6962(strips, 0, ms, nodes, &count);
  for (int j = 0; j < count; ++j) unit_leaf_digest(nodes + 64 * (size_t)j, leaves + 32 * (size_t)(ms + j));
  call_key(salt, forward, position, weight, da, key);
  uint32_t kw[8];
  for (int i = 0; i < 8; ++i) kw[i] = key[4 * i] | (key[4 * i + 1] << 8) | (key[4 * i + 2] << 16) | ((uint32_t)key[4 * i + 3] << 24);
  unit_leaf(kw, 8, 1, leaves + 32 * (size_t)(2 * ms - 1));
}

__global__ void k_x(const int8_t* a, int m, int k, const uint8_t* key, const uint32_t* rowids, const uint32_t* pi, int8_t* e,
                    uint8_t* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  e1_row(key, rowids[i], k, e + (size_t)i * k);
  form_x_row(a + (size_t)i * k, e + (size_t)i * k, pi, k, x + (size_t)i * 3 * k);
}

__global__ void k_y(const int8_t* b, int n, int k, const uint8_t* salt, uint64_t weight, const uint32_t* colids,
                    const uint32_t* inv, int8_t* f, uint8_t* y, uint32_t* c0) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  f1_col(salt, weight, colids[j], k, f + (size_t)j * k);
  c0[j] = form_y_col(b + (size_t)j * k, f + (size_t)j * k, inv, k, y + (size_t)j * 3 * k);
}

// One thread per strip unit: the weight side's ns Y units into ``yleaves``, then the call's ms form_x units.
__global__ void k_form_leaves(int ms, int ns, int k3, const uint8_t* x, const uint8_t* y, const uint32_t* c0, uint8_t* leaves,
                              uint8_t* yleaves) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ms + ns) return;
  Sha256 h;
  unit_leaf_begin(h);
  if (i < ns) {
    unit_leaf_bytes(h, y + (size_t)i * D * k3, (size_t)D * k3);
    for (int c = 0; c < D; ++c) sha256_u32le(h, c0[i * D + c]);
    sha256_final(h, yleaves + 32 * (size_t)i);
  } else {
    unit_leaf_bytes(h, x + (size_t)(i - ns) * D * k3, (size_t)D * k3);
    sha256_final(h, leaves + 32 * (size_t)(2 * ms + i - ns));
  }
}

// A block per tile of the chunk from ``tile0``, a thread per output (r, c): every checkpoint into ck[tile][t][r][c].
__global__ void k_tiles(int ns, int steps, int k3, const uint8_t* x, const uint8_t* y, const uint32_t* c0, size_t tile0,
                        uint32_t* ck) {
  size_t tile = tile0 + blockIdx.x;
  int rs = (int)(tile / ns), cs = (int)(tile % ns), r = threadIdx.x / D, c = threadIdx.x % D;
  tile_entry(x + (size_t)(rs * D + r) * k3, y + (size_t)(cs * D + c) * k3, c0[cs * D + c], steps,
             ck + (size_t)blockIdx.x * steps * D * D + r * D + c, D * D);
}

__global__ void k_tile_leaves(size_t count, int steps, const uint32_t* ck, size_t tile0, uint8_t* leaves) {
  size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (j < count) unit_leaf(ck + j * steps * D * D, (size_t)steps * D * D, 1, leaves + 32 * (tile0 + j));
}

__global__ void k_dequant(size_t count, int ns, int n, int steps, const uint32_t* ck, size_t tile0, const uint32_t* s,
                          const uint32_t* t, const uint16_t* bias, uint16_t* y, uint8_t* leaves) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= count * D * D) return;
  size_t j = i / (D * D), tile = tile0 + j;
  int e = (int)(i % (D * D)), r = e / D, c = e % D;
  int row = (int)(tile / ns) * D + r, col = (int)(tile % ns) * D + c;
  uint16_t v = dequant(ck[j * steps * D * D + (size_t)(steps - 1) * D * D + e], s[row], t[col], bias[col]);
  y[(size_t)row * n + col] = v;
  uint32_t w = v;
  unit_leaf(&w, 1, 1, leaves + 32 * (tile * D * D + e));
}

__global__ void k_units_level(const uint8_t* in, size_t count, uint8_t* out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (2 * i + 1 < count) {
    units_node(in + 64 * i, in + 64 * i + 32, out + 32 * i);
  } else if (2 * i + 1 == count) {
    for (int b = 0; b < 32; ++b) out[32 * i + b] = in[64 * i + b];
  }
}

// `ncp2_host.cpp`'s `ncp2_commit` on the device, and the call's units root; ``ms_out`` is the device time from the first
// kernel to the root, the inputs already resident. Returns 0, 1 for a shape outside the strips, or a CUDA error code.
extern "C" int ncp2_commit(int m, int k, int n, const int8_t* a, const uint32_t* s, const int8_t* b, const uint32_t* t,
                           const uint16_t* bias, const uint8_t* salt, uint64_t forward, uint64_t position, uint64_t weight,
                           const uint32_t* rowids, const uint32_t* colids, const uint32_t* pi, const uint32_t* inv,
                           uint8_t* leaves, uint8_t* yleaves, uint16_t* y_out, uint8_t* root, float* ms_out) {
  if (m <= 0 || n <= 0 || k <= 0 || m % D || n % D || (3 * k) % D || m / D > (1 << 19)) return 1;
  const int ms = m / D, ns = n / D, steps = 3 * k / D, k3 = 3 * k;
  const size_t tiles = (size_t)ms * ns, population = 3 * (size_t)ms + tiles + tiles * D * D;
  const size_t tile_base = 3 * (size_t)ms, deq_base = tile_base + tiles;
  Dev<int8_t> da, db, de, df;
  Dev<uint32_t> ds, dt, drow, dcol, dpi, dinv, dc0, dck;
  Dev<uint16_t> dbias, dy;
  Dev<uint8_t> dsalt, rows, strips, nodes, key, x, y, lv, lv2, lv3, ylv;
  CK(da.put(a, (size_t)m * k));
  CK(db.put(b, (size_t)n * k));
  CK(ds.put(s, m));
  CK(dt.put(t, n));
  CK(dbias.put(bias, n));
  CK(dsalt.put(salt, 32));
  CK(drow.put(rowids, m));
  CK(dcol.put(colids, n));
  CK(dpi.put(pi, k));
  CK(dinv.put(inv, k));
  CK(de.alloc((size_t)m * k));
  CK(df.alloc((size_t)n * k));
  CK(rows.alloc((size_t)m * 64));
  CK(strips.alloc((size_t)ms * 64));
  CK(nodes.alloc((size_t)(ms > 1 ? ms - 1 : 1) * 64));
  CK(key.alloc(32));
  CK(x.alloc((size_t)m * k3));
  CK(y.alloc((size_t)n * k3));
  CK(dc0.alloc(n));
  CK(dy.alloc((size_t)m * n));
  CK(lv.alloc(population * 32));
  CK(ylv.alloc((size_t)ns * 32));
  CK(lv2.alloc((population + 1) / 2 * 32));
  CK(lv3.alloc(((population + 1) / 2 + 1) / 2 * 32));
  const size_t per_tile = (size_t)steps * D * D, chunk = (256u << 20) / (per_tile * 4) ? (256u << 20) / (per_tile * 4) : 1;
  CK(dck.alloc((chunk < tiles ? chunk : tiles) * per_tile));

  CK(cudaDeviceSetLimit(cudaLimitStackSize, 2048));
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  k_row_digests<<<blocks(m, 64), 64>>>(da.p, m, k, rows.p);
  k_strips<<<blocks(ms, 64), 64>>>(rows.p, ms, strips.p, lv.p);
  k_tree_key<<<1, 1>>>(strips.p, ms, nodes.p, dsalt.p, forward, position, weight, key.p, lv.p);
  k_x<<<blocks(m, 64), 64>>>(da.p, m, k, key.p, drow.p, dpi.p, de.p, x.p);
  k_y<<<blocks(n, 64), 64>>>(db.p, n, k, dsalt.p, weight, dcol.p, dinv.p, df.p, y.p, dc0.p);
  k_form_leaves<<<blocks(ms + ns, 64), 64>>>(ms, ns, k3, x.p, y.p, dc0.p, lv.p, ylv.p);
  for (size_t tile0 = 0; tile0 < tiles; tile0 += chunk) {
    size_t count = tiles - tile0 < chunk ? tiles - tile0 : chunk;
    k_tiles<<<(unsigned)count, D * D>>>(ns, steps, k3, x.p, y.p, dc0.p, tile0, dck.p);
    k_tile_leaves<<<blocks(count, 64), 64>>>(count, steps, dck.p, tile0, lv.p + 32 * tile_base);
    k_dequant<<<blocks(count * D * D, 128), 128>>>(count, ns, n, steps, dck.p, tile0, ds.p, dt.p, dbias.p, dy.p, lv.p + 32 * deq_base);
  }
  const uint8_t* in = lv.p;
  uint8_t* out = lv2.p;
  for (size_t count = population; count > 1; count = (count + 1) / 2) {
    k_units_level<<<blocks((count + 1) / 2, 128), 128>>>(in, count, out);
    in = out;
    out = out == lv2.p ? lv3.p : lv2.p;
  }
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  CK(cudaGetLastError());
  CK(cudaEventElapsedTime(ms_out, t0, t1));
  CK(cudaMemcpy(leaves, lv.p, population * 32, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(yleaves, ylv.p, (size_t)ns * 32, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(root, in, 32, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(y_out, dy.p, (size_t)m * n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return 0;
}

// ---- throughput ------------------------------------------------------------------------------------------------------------

// Each thread chains ``iters`` compressions (0 SHA-256, 1 SHA-512) or permutations (2 Keccak-p[1600, 24], 3 Keccak-p[1600,
// 12]) in registers: the hash's ceiling, with no memory traffic.
template <int KIND>
__global__ void k_ceiling(int iters, uint64_t* sink) {
  uint64_t seed = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  uint64_t acc = 0;
  if (KIND == 0) {
    uint32_t h[8], w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = (uint32_t)(seed * 0x9E3779B97F4A7C15ull >> (4 * i));
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = (uint32_t)seed + i;
    for (int it = 0; it < iters; ++it) {
      sha256_compress(h, w);
      w[0] ^= h[0];
    }
    acc = ((uint64_t)h[0] << 32 | h[1]) ^ ((uint64_t)h[2] << 32 | h[3]) ^ ((uint64_t)h[4] << 32 | h[5]) ^ ((uint64_t)h[6] << 32 | h[7]);
  } else if (KIND == 1) {
    uint64_t h[8], w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = seed * 0x9E3779B97F4A7C15ull + i;
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = seed + i;
    for (int it = 0; it < iters; ++it) {
      sha512_compress(h, w);
      w[0] ^= h[0];
    }
    acc = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7];
  } else {
    uint64_t st[25];
#pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = seed + i;
    for (int it = 0; it < iters; ++it) keccak_p<KIND == 2 ? 24 : 12>(st);
#pragma unroll
    for (int i = 0; i < 25; ++i) acc ^= st[i];
  }
  if (acc == 0x5A5A5A5A5A5A5A5Aull) sink[0] = acc;       // never true, but the compiler cannot know: the work is kept
}

static void ceiling(int kind, int grid, int threads, int iters, uint64_t* sink) {
  if (kind == 0) k_ceiling<0><<<grid, threads>>>(iters, sink);
  else if (kind == 1) k_ceiling<1><<<grid, threads>>>(iters, sink);
  else if (kind == 2) k_ceiling<2><<<grid, threads>>>(iters, sink);
  else k_ceiling<3><<<grid, threads>>>(iters, sink);
}

// Bytes one compression or permutation absorbs: the block, or the sponge's rate.
static const double CEILING_BYTES[4] = {64, 128, 136, 168};

extern "C" int ncp2_ceiling(int kind, int grid, int threads, int iters, double* bytes_per_s) {
  Dev<uint64_t> sink;
  CK(sink.alloc(1));
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  if (kind < 0 || kind > 3) return 1;
  ceiling(kind, grid, threads, 2, sink.p);
  CK(cudaEventRecord(t0));
  ceiling(kind, grid, threads, iters, sink.p);
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, t0, t1));
  *bytes_per_s = CEILING_BYTES[kind] * grid * (double)threads * iters / (ms * 1e-3);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return 0;
}

__global__ void k_fill(uint32_t* w, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) w[i] = (uint32_t)(i * 2654435761u);
}

// ``units`` word leaves of ``words`` each, word i of unit u at w[i · units + u] (coalesced): kind 0 SHA-256, 1 SHA-512,
// 3 TurboSHAKE128.
__global__ void k_leaves(int kind, const uint32_t* w, size_t units, uint32_t words, uint8_t* out) {
  size_t u = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (u >= units) return;
  if (kind == 0) unit_leaf(w + u, words, units, out + 32 * u);
  else if (kind == 1) unit_leaf_sha512(w + u, words, units, out + 64 * u);
  else unit_leaf_turbo(w + u, words, units, out + 32 * u);
}

// The mean time of ``reps`` launches hashing ``units`` leaves of ``words`` words from device memory.
extern "C" int ncp2_leaf_time(int kind, uint64_t units, uint32_t words, int reps, float* ms_out) {
  Dev<uint32_t> w;
  Dev<uint8_t> out;
  CK(w.alloc((size_t)units * words));
  CK(out.alloc((size_t)units * 64));
  k_fill<<<blocks((size_t)units * words, 256), 256>>>(w.p, (size_t)units * words);
  k_leaves<<<blocks(units, 128), 128>>>(kind, w.p, units, words, out.p);
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  for (int r = 0; r < reps; ++r) k_leaves<<<blocks(units, 128), 128>>>(kind, w.p, units, words, out.p);
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, t0, t1));
  *ms_out = ms / reps;
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return 0;
}

__global__ void k_f1(const uint8_t* salt, uint64_t weight, uint32_t n, int k, int8_t* f) {
  uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) f1_col(salt, weight, j, k, f + (size_t)j * k);
}

// The mean time of deriving F₁ for ``n`` columns at ``k``: the per-call XOF a weight of n rows costs.
extern "C" int ncp2_f1_time(uint32_t n, int k, int reps, float* ms_out) {
  Dev<uint8_t> salt;
  Dev<int8_t> f;
  uint8_t zeros[32] = {0};
  CK(salt.put(zeros, 32));
  CK(f.alloc((size_t)n * k));
  k_f1<<<blocks(n, 64), 64>>>(salt.p, 7, n, k, f.p);
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  for (int r = 0; r < reps; ++r) k_f1<<<blocks(n, 64), 64>>>(salt.p, 7, n, k, f.p);
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, t0, t1));
  *ms_out = ms / reps;
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return 0;
}
