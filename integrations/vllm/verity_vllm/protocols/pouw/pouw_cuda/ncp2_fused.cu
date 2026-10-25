// `ncp-v2`'s serving kernel (`pouw_device.fused`): one call committed on the device from resident inputs, the served y its
// dequantization output. Every function is `ncp2_fast.cuh`'s, which `ncp2_host.cpp` runs on the CPU against `pouw_native`.
//   - tiles: a thread per 16 × 16 tile, its 256 running sums in shared memory; each 16-deep step's row of 16 words is one
//     SHA-256 block of the tile's word leaf, hashed as it is made (dp4a for the products);
//   - the key (SHA-512 digests, the RFC 6962 tree, K), E₁ and F₁ (SHAKE256 a block at a time), X and Y per call;
//   - the form strips' leaves on a second stream beside the tiles; Y's leaves, which depend only on the weight, the salt and
//     the column ids, cached per weight when asked (flag 4);
//   - the dequantization and its word leaves, and the units tree, a thread per node.
// Flag 1 hashes; without it every hash and XOF is left out and the noise is a counter-based stand-in (the no-hash control).
// Flag 2 times the tile kernel with CUDA events, read by `ncp2_fused_tile_ms` without a synchronization per call. Flag 8
// swaps the tiles' word leaves for the low-byte variant (`tile_fused_lowbyte`), for timing only: its γ is unproved.
#include <cuda_runtime.h>

#include <vector>

#include "ncp2_fast.cuh"

using namespace ncp2;

#define CK(x)                              \
  do {                                     \
    cudaError_t e_ = (x);                  \
    if (e_ != cudaSuccess) return (int)e_; \
  } while (0)

struct Salt {
  uint8_t b[32];
};

static unsigned nblocks(size_t n, unsigned per) { return (unsigned)((n + per - 1) / per); }

// ---- kernels ------------------------------------------------------------------------------------------------------------

__global__ void kf_rows(const int8_t* a, int m, int k, uint8_t* rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) row_digest_fast(a + (size_t)i * k, k, rows + 64 * (size_t)i);
}

__global__ void kf_strips(const uint8_t* rows, int ms, uint8_t* strips, uint8_t* leaves) {
  int st = blockIdx.x * blockDim.x + threadIdx.x;
  if (st >= ms) return;
  strip_digest_fast(rows + (size_t)st * D * 64, strips + 64 * (size_t)st);
  leaf_digest_fast(strips + 64 * (size_t)st, leaves + 32 * (size_t)st);
}

__global__ void kf_tree_key(const uint8_t* strips, int ms, uint8_t* nodes, Salt salt, uint64_t forward, uint64_t position,
                            uint64_t weight, uint8_t* key) {
  int count = 0;
  const uint8_t* da = rfc6962_fast(strips, 0, ms, nodes, &count);
  call_key_fast(salt.b, forward, position, weight, da, key);
}

// Its own kernel: inlined into `kf_tree_key`, the key's leaf takes nvcc's front end minutes.
__global__ void kf_key_leaf(const uint8_t* key, int ms, uint8_t* leaves) {
  uint32_t kw[8];
  for (int i = 0; i < 8; ++i) kw[i] = key[4 * i] | (key[4 * i + 1] << 8) | (key[4 * i + 2] << 16) | ((uint32_t)key[4 * i + 3] << 24);
  unit_leaf(kw, 8, 1, leaves + 32 * (size_t)(2 * ms - 1));
}

__global__ void kf_node_leaves(const uint8_t* nodes, int count, int ms, uint8_t* leaves) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < count) leaf_digest_fast(nodes + 64 * (size_t)j, leaves + 32 * (size_t)(ms + j));
}

template <bool HASH>
__global__ void kf_x(const int8_t* a, int m, int k, const uint8_t* key, const uint32_t* rowids, const uint32_t* pi, int8_t* e,
                     uint8_t* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  if (HASH) e1_row_fast(key, rowids[i], k, e + (size_t)i * k);
  else standin_noise(rowids[i], 0xE1u, k, e + (size_t)i * k);
  form_x_row(a + (size_t)i * k, e + (size_t)i * k, pi, k, x + (size_t)i * 3 * k);
}

template <bool HASH>
__global__ void kf_y(const int8_t* b, int n, int k, Salt salt, uint64_t weight, const uint32_t* colids, const uint32_t* inv,
                     int8_t* f, uint8_t* y, uint32_t* c0) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  if (HASH) f1_col_fast(salt.b, weight, colids[j], k, f + (size_t)j * k);
  else standin_noise(colids[j], 0xF1u ^ (uint32_t)weight, k, f + (size_t)j * k);
  c0[j] = form_y_col(b + (size_t)j * k, f + (size_t)j * k, inv, k, y + (size_t)j * 3 * k);
}

// Threads [0, ms): the call's form_x units; then, when ``with_y``, [ms, ms + ns): the weight side's Y units into ``yleaves``.
__global__ void kf_form_leaves(int ms, int ns, int with_y, int k3, const uint8_t* x, const uint8_t* y, const uint32_t* c0,
                               uint8_t* leaves, uint8_t* yleaves) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ms + (with_y ? ns : 0)) return;
  WordLeaf wl;
  wl_init(wl);
  if (i < ms) {
    wl_bytes(wl, x + (size_t)i * D * k3, (size_t)D * k3);
    wl_final(wl, leaves + 32 * (size_t)(2 * ms + i));
  } else {
    int cs = i - ms;
    wl_bytes(wl, y + (size_t)cs * D * k3, (size_t)D * k3);
    wl_block(wl, c0 + (size_t)cs * D);
    wl_final(wl, yleaves + 32 * (size_t)cs);
  }
}

struct SmemRun {
  uint32_t* base;
  __device__ __forceinline__ uint32_t& operator()(int j) const { return base[j * 32]; }
};

// LEAF 0: no leaf (the no-hash control); 1: the word leaf (committed); 2: the low-byte variant (timed only; γ unproved).
template <int LEAF>
__global__ void __launch_bounds__(32) kf_tiles(int ns, size_t tiles, int steps, int k3, int n, const uint8_t* x,
                                               const uint8_t* y, const uint32_t* c0, uint8_t* tile_leaves, uint32_t* z) {
  extern __shared__ uint32_t sm[];
  const int lane = threadIdx.x;
  const size_t tile = (size_t)blockIdx.x * 32 + lane;
  if (tile >= tiles) return;
  const int rs = (int)(tile / ns), cs = (int)(tile % ns);
  SmemRun run{sm + lane};
  for (int j = 0; j < D * D; ++j) run(j) = c0[cs * D + (j & 15)];
  uint8_t leaf[32];
  if (LEAF == 2) tile_fused_lowbyte(x + (size_t)rs * D * k3, y + (size_t)cs * D * k3, k3, steps, run, leaf);
  else tile_fused<LEAF == 1>(x + (size_t)rs * D * k3, y + (size_t)cs * D * k3, k3, steps, run, leaf);
  if (LEAF)
    for (int i = 0; i < 32; ++i) tile_leaves[32 * tile + i] = leaf[i];
  for (int j = 0; j < D * D; ++j) z[(size_t)(rs * D + (j >> 4)) * n + cs * D + (j & 15)] = run(j);
}

template <bool HASH>
__global__ void kf_dequant(int m, int n, const uint32_t* z, const uint32_t* s, const uint32_t* t, const uint16_t* bias,
                           uint16_t* y, uint8_t* leaves) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)m * n) return;
  const int row = (int)(i / n), col = (int)(i % n);
  const uint16_t v = dequant(z[i], s[row], t[col], bias[col]);
  y[i] = v;
  if (HASH) {
    const int ns = n / D;
    const size_t u = ((size_t)(row / D) * ns + col / D) * D * D + (row % D) * D + col % D;
    leaf_word1(v, leaves + 32 * u);
  }
}

__device__ __forceinline__ void load_digest(const uint8_t* p, uint32_t w[8]) {
  const uint4 a = *(const uint4*)p, b = *(const uint4*)(p + 16);
  w[0] = bswap32(a.x); w[1] = bswap32(a.y); w[2] = bswap32(a.z); w[3] = bswap32(a.w);
  w[4] = bswap32(b.x); w[5] = bswap32(b.y); w[6] = bswap32(b.z); w[7] = bswap32(b.w);
}

__global__ void kf_units_level(const uint8_t* in, size_t count, uint8_t* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (2 * i + 1 < count) {
    uint32_t L[8], R[8], h[8];
    load_digest(in + 64 * i, L);
    load_digest(in + 64 * i + 32, R);
    units_node_fast(L, R, h);
    uint4* o = (uint4*)(out + 32 * i);
    o[0] = make_uint4(bswap32(h[0]), bswap32(h[1]), bswap32(h[2]), bswap32(h[3]));
    o[1] = make_uint4(bswap32(h[4]), bswap32(h[5]), bswap32(h[6]), bswap32(h[7]));
  } else if (2 * i + 1 == count) {
    const uint4* src = (const uint4*)(in + 64 * i);
    uint4* o = (uint4*)(out + 32 * i);
    o[0] = src[0];
    o[1] = src[1];
  }
}

// ---- the workspace, the Y leaf cache and the tile timings ----------------------------------------------------------------

struct Grow {
  void* p = nullptr;
  size_t cap = 0;
  cudaError_t need(size_t bytes) {
    if (bytes <= cap) return cudaSuccess;
    if (p) cudaFree(p);
    cap = bytes + bytes / 4;
    return cudaMalloc(&p, cap);
  }
  template <class T>
  T* as() const { return (T*)p; }
};

static struct {
  Grow rows, strips, nodes, key, e, f, x, y, c0, z, leaves, yleaves, lv2, lv3, root;
  const uint8_t* ylast = nullptr;  // the last call's Y leaves: ``yleaves`` or its weight's cached copy
  cudaStream_t aux = nullptr;
  cudaEvent_t xy = nullptr, formed = nullptr;
} W;

struct YCache {
  uint64_t weight;
  int n, k;
  uint8_t salt[32];
  uint8_t* leaves;
};
static std::vector<YCache> ycache;
static std::vector<std::pair<cudaEvent_t, cudaEvent_t>> tile_events;

static YCache* ycache_find(uint64_t weight, int n, int k, const uint8_t* salt) {
  for (auto& c : ycache) {
    bool same = c.weight == weight && c.n == n && c.k == k;
    for (int i = 0; same && i < 32; ++i) same = c.salt[i] == salt[i];
    if (same) return &c;
  }
  return nullptr;
}

// ---- the call ----------------------------------------------------------------------------------------------------------------

// One call from device inputs on ``stream``: y (m × n BF16 encodings) into ``y_out``; with flag 1, every unit leaf of the call
// and its units root in the workspace, and its weight side's Y unit leaves, copied to ``root_host`` / ``leaves_host`` /
// ``yleaves_host`` when they are given (host memory; the stream is then synchronized). Returns 0, 1 for a shape outside the
// strips, or a CUDA error code.
extern "C" int ncp2_fused(int m, int k, int n, const int8_t* a, const uint32_t* s, const int8_t* b, const uint32_t* t,
                          const uint16_t* bias, const uint8_t* salt_host, uint64_t forward, uint64_t position, uint64_t weight,
                          const uint32_t* rowids, const uint32_t* colids, const uint32_t* pi, const uint32_t* inv, int flags,
                          uint64_t stream_handle, uint16_t* y_out, uint8_t* root_host, uint8_t* leaves_host,
                          uint8_t* yleaves_host) {
  if (m <= 0 || n <= 0 || k <= 0 || m % D || n % D || (3 * k) % D || m / D > (1 << 19)) return 1;
  const bool hash = flags & 1, timed = flags & 2, cache = flags & 4, lowbyte = flags & 8;
  cudaStream_t st = (cudaStream_t)stream_handle;
  const int ms = m / D, ns = n / D, steps = 3 * k / D, k3 = 3 * k;
  const size_t tiles = (size_t)ms * ns, population = 3 * (size_t)ms + tiles + tiles * D * D;
  const size_t tile_base = 3 * (size_t)ms, deq_base = tile_base + tiles;
  Salt salt;
  for (int i = 0; i < 32; ++i) salt.b[i] = salt_host[i];
  if (!W.aux) {
    CK(cudaStreamCreateWithFlags(&W.aux, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&W.xy, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&W.formed, cudaEventDisableTiming));
    CK(cudaFuncSetAttribute(kf_tiles<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * D * D * 4));
    CK(cudaFuncSetAttribute(kf_tiles<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * D * D * 4));
    CK(cudaFuncSetAttribute(kf_tiles<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * D * D * 4));
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 2048));
  }
  CK(W.e.need((size_t)m * k));
  CK(W.f.need((size_t)n * k));
  CK(W.x.need((size_t)m * k3));
  CK(W.y.need((size_t)n * k3));
  CK(W.c0.need((size_t)n * 4));
  CK(W.z.need((size_t)m * n * 4));
  CK(W.key.need(32));
  if (hash) {
    CK(W.rows.need((size_t)m * 64));
    CK(W.strips.need((size_t)ms * 64));
    CK(W.nodes.need((size_t)(ms > 1 ? ms - 1 : 1) * 64));
    CK(W.leaves.need(population * 32));
    CK(W.yleaves.need((size_t)ns * 32));
    CK(W.lv2.need((population + 1) / 2 * 32));
    CK(W.lv3.need(((population + 1) / 2 + 1) / 2 * 32 + 32));
    CK(W.root.need(32));
  }
  uint8_t* lv = W.leaves.as<uint8_t>();
  YCache* yc = (hash && cache) ? ycache_find(weight, n, k, salt.b) : nullptr;

  if (hash) {
    kf_rows<<<nblocks(m, 64), 64, 0, st>>>(a, m, k, W.rows.as<uint8_t>());
    kf_strips<<<nblocks(ms, 64), 64, 0, st>>>(W.rows.as<uint8_t>(), ms, W.strips.as<uint8_t>(), lv);
    kf_tree_key<<<1, 1, 0, st>>>(W.strips.as<uint8_t>(), ms, W.nodes.as<uint8_t>(), salt, forward, position, weight,
                                 W.key.as<uint8_t>());
    kf_key_leaf<<<1, 1, 0, st>>>(W.key.as<uint8_t>(), ms, lv);
    if (ms > 1) kf_node_leaves<<<nblocks(ms - 1, 64), 64, 0, st>>>(W.nodes.as<uint8_t>(), ms - 1, ms, lv);
    kf_x<true><<<nblocks(m, 64), 64, 0, st>>>(a, m, k, W.key.as<uint8_t>(), rowids, pi, W.e.as<int8_t>(), W.x.as<uint8_t>());
    kf_y<true><<<nblocks(n, 64), 64, 0, st>>>(b, n, k, salt, weight, colids, inv, W.f.as<int8_t>(), W.y.as<uint8_t>(),
                                             W.c0.as<uint32_t>());
    CK(cudaEventRecord(W.xy, st));
    CK(cudaStreamWaitEvent(W.aux, W.xy, 0));
    kf_form_leaves<<<nblocks(ms + (yc ? 0 : ns), 32), 32, 0, W.aux>>>(ms, ns, yc ? 0 : 1, k3, W.x.as<uint8_t>(),
                                                                     W.y.as<uint8_t>(), W.c0.as<uint32_t>(), lv,
                                                                     W.yleaves.as<uint8_t>());
    W.ylast = yc ? yc->leaves : W.yleaves.as<uint8_t>();
    if (!yc && cache) {
      YCache c{weight, n, k, {}, nullptr};
      for (int i = 0; i < 32; ++i) c.salt[i] = salt.b[i];
      CK(cudaMalloc((void**)&c.leaves, 32 * (size_t)ns));
      CK(cudaMemcpyAsync(c.leaves, W.yleaves.p, 32 * (size_t)ns, cudaMemcpyDeviceToDevice, W.aux));
      ycache.push_back(c);
    }
    CK(cudaEventRecord(W.formed, W.aux));
  } else {
    kf_x<false><<<nblocks(m, 64), 64, 0, st>>>(a, m, k, nullptr, rowids, pi, W.e.as<int8_t>(), W.x.as<uint8_t>());
    kf_y<false><<<nblocks(n, 64), 64, 0, st>>>(b, n, k, salt, weight, colids, inv, W.f.as<int8_t>(), W.y.as<uint8_t>(),
                                              W.c0.as<uint32_t>());
  }

  cudaEvent_t t0 = nullptr, t1 = nullptr;
  if (timed) {
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0, st));
  }
  const size_t smem = 32 * D * D * 4;
  if (hash && lowbyte)
    kf_tiles<2><<<nblocks(tiles, 32), 32, smem, st>>>(ns, tiles, steps, k3, n, W.x.as<uint8_t>(), W.y.as<uint8_t>(),
                                                     W.c0.as<uint32_t>(), lv + 32 * tile_base, W.z.as<uint32_t>());
  else if (hash)
    kf_tiles<1><<<nblocks(tiles, 32), 32, smem, st>>>(ns, tiles, steps, k3, n, W.x.as<uint8_t>(), W.y.as<uint8_t>(),
                                                     W.c0.as<uint32_t>(), lv + 32 * tile_base, W.z.as<uint32_t>());
  else
    kf_tiles<0><<<nblocks(tiles, 32), 32, smem, st>>>(ns, tiles, steps, k3, n, W.x.as<uint8_t>(), W.y.as<uint8_t>(),
                                                     W.c0.as<uint32_t>(), nullptr, W.z.as<uint32_t>());
  if (timed) {
    CK(cudaEventRecord(t1, st));
    tile_events.emplace_back(t0, t1);
  }
  if (hash)
    kf_dequant<true><<<nblocks((size_t)m * n, 128), 128, 0, st>>>(m, n, W.z.as<uint32_t>(), s, t, bias, y_out, lv + 32 * deq_base);
  else
    kf_dequant<false><<<nblocks((size_t)m * n, 128), 128, 0, st>>>(m, n, W.z.as<uint32_t>(), s, t, bias, y_out, nullptr);
  CK(cudaGetLastError());
  if (!hash) return 0;

  CK(cudaStreamWaitEvent(st, W.formed, 0));
  const uint8_t* in = lv;
  uint8_t* out = W.lv2.as<uint8_t>();
  for (size_t count = population; count > 1; count = (count + 1) / 2) {
    kf_units_level<<<nblocks((count + 1) / 2, 128), 128, 0, st>>>(in, count, out);
    in = out;
    out = out == W.lv2.as<uint8_t>() ? W.lv3.as<uint8_t>() : W.lv2.as<uint8_t>();
  }
  CK(cudaGetLastError());
  CK(cudaMemcpyAsync(W.root.p, in, 32, cudaMemcpyDeviceToDevice, st));
  if (root_host || leaves_host || yleaves_host) {
    if (root_host) CK(cudaMemcpyAsync(root_host, in, 32, cudaMemcpyDeviceToHost, st));
    if (leaves_host) CK(cudaMemcpyAsync(leaves_host, lv, population * 32, cudaMemcpyDeviceToHost, st));
    if (yleaves_host) CK(cudaMemcpyAsync(yleaves_host, W.ylast, 32 * (size_t)ns, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
  }
  return 0;
}

// The tile kernel's time summed over the calls timed since the last read (flag 2), and their count; the events are released.
extern "C" int ncp2_fused_tile_ms(double* total_ms, uint64_t* calls) {
  double total = 0;
  for (auto& e : tile_events) {
    CK(cudaEventSynchronize(e.second));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e.first, e.second));
    total += ms;
    cudaEventDestroy(e.first);
    cudaEventDestroy(e.second);
  }
  *total_ms = total;
  *calls = tile_events.size();
  tile_events.clear();
  return 0;
}

extern "C" int ncp2_fused_reset_cache() {
  for (auto& c : ycache) cudaFree(c.leaves);
  ycache.clear();
  return 0;
}

// ---- host pointers, for the gate ----------------------------------------------------------------------------------------------

template <class T>
struct DevBuf {
  T* p = nullptr;
  ~DevBuf() { if (p) cudaFree(p); }
  cudaError_t put(const T* h, size_t n) {
    cudaError_t e = cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T));
    return e != cudaSuccess ? e : cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice);
  }
  cudaError_t alloc(size_t n) { return cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T)); }
};

// `ncp2_commit`'s signature (host pointers) through `ncp2_fused` on the default stream with ``flags``; ``ms_out`` is the call's
// device time, the inputs already resident.
extern "C" int ncp2_fused_host(int m, int k, int n, const int8_t* a, const uint32_t* s, const int8_t* b, const uint32_t* t,
                               const uint16_t* bias, const uint8_t* salt, uint64_t forward, uint64_t position, uint64_t weight,
                               const uint32_t* rowids, const uint32_t* colids, const uint32_t* pi, const uint32_t* inv,
                               uint8_t* leaves, uint8_t* yleaves, uint16_t* y_out, uint8_t* root, float* ms_out, int flags) {
  DevBuf<int8_t> da, db;
  DevBuf<uint32_t> ds, dt, drow, dcol, dpi, dinv;
  DevBuf<uint16_t> dbias, dy;
  CK(da.put(a, (size_t)m * k));
  CK(db.put(b, (size_t)n * k));
  CK(ds.put(s, m));
  CK(dt.put(t, n));
  CK(dbias.put(bias, n));
  CK(drow.put(rowids, m));
  CK(dcol.put(colids, n));
  CK(dpi.put(pi, k));
  CK(dinv.put(inv, k));
  CK(dy.alloc((size_t)m * n));
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0, 0));
  int rc = ncp2_fused(m, k, n, da.p, ds.p, db.p, dt.p, dbias.p, salt, forward, position, weight, drow.p, dcol.p, dpi.p, dinv.p,
                      flags, 0, dy.p, nullptr, nullptr, nullptr);
  if (rc) return rc;
  CK(cudaEventRecord(t1, 0));
  CK(cudaEventSynchronize(t1));
  CK(cudaEventElapsedTime(ms_out, t0, t1));
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (flags & 1) {
    const int ms = m / D, ns = n / D;
    const size_t tiles = (size_t)ms * ns, population = 3 * (size_t)ms + tiles + tiles * D * D;
    CK(cudaMemcpy(leaves, W.leaves.p, population * 32, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(yleaves, W.ylast, 32 * (size_t)ns, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(root, W.root.p, 32, cudaMemcpyDeviceToHost));
  }
  CK(cudaMemcpy(y_out, dy.p, (size_t)m * n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  return 0;
}
