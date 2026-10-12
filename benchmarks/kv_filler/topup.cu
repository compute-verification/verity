// The filler top-up: a host thread of its own reads registered regions of HBM on a low-priority stream, chunk after
// chunk, each chunk with its receipts, so that a window's reads approach R·T while a serving process runs beside it and
// its Python never waits on the top-up.
//
//   mode -1  read only: every 16 bytes loaded once and XOR-folded (the floor of a receipt's cost)
//   mode  0  gf31-poly/v1 over 2,056-byte blocks, the bandwidth audit's uncertified fast check
//   mode  2  ro-blake3/v1 over 2,056-byte blocks, the bandwidth audit's certified receipt, with its leaves and tree
//   mode  3  sk16: the cost of a sketch-then-hash receipt over the same blocks (ra-sketch-tag's shape, not its format):
//            each block's 2,056 bytes as int8 against 16 rows of ±1 signs (dp4a), the 16 int32 sums hashed by one keyed
//            BLAKE3 compression under the block's index, then ro-blake3/v1's leaves and tree over those 32-byte digests
//
// The receipts are the bandwidth audit's own kernels (`../bandwidth/tagger.cu`, included here): a chunk is one tag pass
// over `chunk_bytes / 2056` blocks of a region under the chunk's seed (the run's seed with its first 8 bytes XORed with
// the chunk's number), its root copied to pinned memory once the chunk ends.  A chunk is launched as many CTAs of
// microseconds each (read: one CTA a `tile_bytes` tile; ro-blake3/v1: 64 blocks a CTA), so an inference kernel queued
// on a higher-priority stream gets an SM as soon as a tile ends.  `depth` chunks are kept in flight.
//
// Each chunk's start and end are CUDA events, read back as milliseconds since the start event, so a window's filler
// bytes come from the GPU's clock, not the host's.  `tu_pause(1)` stops launching new chunks (the in-flight ones
// finish): the engine-driven, idle-only top-up.
//
// Built as libtopup.so (run.sh).  C ABI at the end.

#include "../bandwidth/tagger.cu"

#include <atomic>
#include <mutex>
#include <thread>

namespace tu {

constexpr int MODE_READ = -1, MODE_SK = 3;
constexpr int SK_ROWS = 16, SK_WORDS = (int)(bw::BLOCK_BYTES / 4), SK_G = 2, SK_WARPS = 8, SK_KS = 17;
constexpr size_t SK_SMEM = sizeof(uint32_t) * (SK_WORDS * SK_ROWS + SK_WARPS * 32 * SK_KS);
static const char SK_NAME[] = "verity/kv-filler/sk16-proxy/v0";

struct Region {
  const uint8_t* p;
  uint64_t bytes;
};

struct Rec {
  double t0_ms, t1_ms;  // the chunk's start and end events, since the start event
  uint64_t bytes;
  uint32_t region, root4;  // root4: the first 4 bytes of the chunk's root (mode >= 0), else 0
  uint64_t chunk;
};

// Tiles of tile16 16-byte words, CTA b taking tiles b, b + gridDim.x, ...: one CTA a tile at full grid, or a few
// long-lived CTAs (the throttled top-up).
__global__ void __launch_bounds__(256) tile_read(const uint4* __restrict__ p, uint64_t n16, uint32_t tile16,
                                                 uint32_t* out) {
  uint32_t acc = 0;
  for (uint64_t base = (uint64_t)blockIdx.x * tile16; base < n16; base += (uint64_t)gridDim.x * tile16) {
    const uint64_t end = base + tile16 < n16 ? base + tile16 : n16;
    for (uint64_t i = base + threadIdx.x; i < end; i += 256) {
      uint4 v = __ldcs(p + i);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, d);
  if ((threadIdx.x & 31) == 0 && acc == 0x9e3779b9u) atomicXor(out, acc);  // keeps the loads; almost never stores
}

// The ±1 signs: row r's byte 4q + b is bit b of nibble r of the q-th 64-bit SplitMix64 output from the seed's first 8
// bytes, set meaning -1; stored [q][r] as int8x4 words.
static void sk_signs(const uint8_t seed[32], uint32_t* out) {
  uint64_t x = 0;
  for (int k = 0; k < 8; ++k) x |= (uint64_t)seed[k] << (8 * k);
  for (int q = 0; q < SK_WORDS; ++q) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    for (int r = 0; r < SK_ROWS; ++r) {
      uint32_t w = 0;
      for (int b = 0; b < 4; ++b) w |= (uint32_t)((z >> (4 * r + b)) & 1 ? 0xFFu : 0x01u) << (8 * b);
      out[q * SK_ROWS + r] = w;
    }
  }
}

// One warp per 32 blocks, SK_G blocks at a time against each sign word, then a halving reduction that leaves row
// (lane >> 1) & 15 of each block's sums in lanes 2i and 2i + 1; lane b hashes block base + b.
__global__ void __launch_bounds__(SK_WARPS * 32, 2) sk_kernel(const uint64_t* __restrict__ E8, uint64_t count,
                                                           const uint32_t* __restrict__ signs, bw::Key3 key,
                                                           uint32_t* __restrict__ tags) {
  extern __shared__ __align__(16) uint32_t smem[];
  uint32_t* S = smem;
  uint32_t* KS = smem + SK_WORDS * SK_ROWS + (threadIdx.x >> 5) * 32 * SK_KS;
  for (int i = threadIdx.x; i < SK_WORDS * SK_ROWS / 4; i += blockDim.x) ((uint4*)S)[i] = ((const uint4*)signs)[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const uint64_t base = ((uint64_t)blockIdx.x * SK_WARPS + (threadIdx.x >> 5)) * 32;
  if (base >= count) return;
  const int myrow = (lane >> 1) & 15;
#pragma unroll 1
  for (int g0 = 0; g0 < 32; g0 += SK_G) {
    int acc[SK_G][SK_ROWS];
#pragma unroll
    for (int g = 0; g < SK_G; ++g)
#pragma unroll
      for (int r = 0; r < SK_ROWS; ++r) acc[g][r] = 0;
#pragma unroll
    for (int t = 0; t < 9; ++t) {
      const int w = lane + 32 * t;
      if (w < (int)bw::ROW_U64) {
        uint64_t v[SK_G];
#pragma unroll
        for (int g = 0; g < SK_G; ++g) {
          const uint64_t j = base + g0 + g;
          v[g] = j < count ? __ldcs(E8 + j * bw::ROW_U64 + w) : 0;
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint4* sq = (const uint4*)(S + (2 * w + h) * SK_ROWS);
          uint32_t sr[SK_ROWS];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint4 u = sq[i];
            sr[4 * i] = u.x, sr[4 * i + 1] = u.y, sr[4 * i + 2] = u.z, sr[4 * i + 3] = u.w;
          }
#pragma unroll
          for (int g = 0; g < SK_G; ++g) {
            const int x = (int)(uint32_t)(v[g] >> (32 * h));
#pragma unroll
            for (int r = 0; r < SK_ROWS; ++r) acc[g][r] = __dp4a(x, (int)sr[r], acc[g][r]);
          }
        }
      }
    }
#pragma unroll
    for (int g = 0; g < SK_G; ++g) {
      int* a = acc[g];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int send = (lane & 16) ? a[k] : a[k + 8], keep = (lane & 16) ? a[k + 8] : a[k];
        a[k] = keep + __shfl_xor_sync(bw::FULL, send, 16);
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int send = (lane & 8) ? a[k] : a[k + 4], keep = (lane & 8) ? a[k + 4] : a[k];
        a[k] = keep + __shfl_xor_sync(bw::FULL, send, 8);
      }
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int send = (lane & 4) ? a[k] : a[k + 2], keep = (lane & 4) ? a[k + 2] : a[k];
        a[k] = keep + __shfl_xor_sync(bw::FULL, send, 4);
      }
      {
        const int send = (lane & 2) ? a[0] : a[1], keep = (lane & 2) ? a[1] : a[0];
        a[0] = keep + __shfl_xor_sync(bw::FULL, send, 2);
      }
      a[0] += __shfl_xor_sync(bw::FULL, a[0], 1);
      if (!(lane & 1)) KS[(g0 + g) * SK_KS + myrow] = (uint32_t)a[0];
    }
  }
  __syncwarp();
  const uint64_t j = base + lane;
  if (j < count) {
    uint32_t m[16], y[8];
#pragma unroll
    for (int r = 0; r < 16; ++r) m[r] = KS[lane * SK_KS + r];
    bw::b3_compress(key.k, m, (uint32_t)j, 64, bw::B3_KEYED | bw::B3_CHUNK_START | bw::B3_CHUNK_END | bw::B3_ROOT, y);
    uint4* o = (uint4*)(tags + j * 8);
    o[0] = make_uint4(y[0], y[1], y[2], y[3]);
    o[1] = make_uint4(y[4], y[5], y[6], y[7]);
  }
}

// The sk16 pass over `count` blocks of E: digests, then ro-blake3/v1's leaves (under SK_NAME) and tree.
static void sk_pass(const bw::Pass& p, const void* E, const uint32_t* signs, const uint8_t seed[32], uint64_t count,
                    cudaStream_t st) {
  const unsigned grid = (unsigned)((count + 32 * SK_WARPS - 1) / (32 * SK_WARPS));
  sk_kernel<<<grid, SK_WARPS * 32, SK_SMEM, st>>>((const uint64_t*)E, count, signs, bw::b3_key(seed), p.tags);
  const unsigned leaf_grid = (unsigned)(((count + 15) / 16 + 127) / 128);
  bw::b3_leaves<<<leaf_grid, 128, 0, st>>>(p.tags, p.blocks, count, bw::prefix_midstate(bw::LEAF_DOMAIN, SK_NAME, seed),
                                           p.nodes);
  bw::launch_tree(p, st);
}

// The host's digest of block E_j (2,056 bytes) under sk16: the gate's reference.
static void sk_host(const uint8_t* Ej, uint64_t j, const uint32_t* signs, const bw::Key3& key, uint32_t y[8]) {
  uint32_t m[16];
  for (int r = 0; r < SK_ROWS; ++r) {
    int64_t sum = 0;
    for (int q = 0; q < SK_WORDS; ++q)
      for (int b = 0; b < 4; ++b)
        sum += (int64_t)(int8_t)Ej[4 * q + b] * (int8_t)(uint8_t)(signs[q * SK_ROWS + r] >> (8 * b));
    m[r] = (uint32_t)(int32_t)sum;
  }
  bw::b3_compress(key.k, m, (uint32_t)j, 64, bw::B3_KEYED | bw::B3_CHUNK_START | bw::B3_CHUNK_END | bw::B3_ROOT, y);
}

struct State {
  int device = 0, mode = MODE_READ, depth = 2, priority = 0, ctas = 0;
  uint64_t chunk_bytes = 1ull << 28, tile_bytes = 1ull << 16;
  std::vector<Region> regions;
  uint8_t seed[32];
  cudaStream_t s = nullptr;
  cudaEvent_t start = nullptr;
  std::vector<cudaEvent_t> e0, e1;
  std::vector<void*> ws;
  std::vector<uint64_t> slot_bytes, slot_chunk;
  std::vector<uint32_t> slot_region;
  std::vector<int> busy;
  uint8_t* roots_h = nullptr;  // pinned, depth × 64
  uint32_t* xr = nullptr;
  uint32_t* signs = nullptr;  // sk16's, on the device
  std::thread th;
  std::atomic<int> run{0}, paused{0}, err{0};
  std::atomic<uint64_t> bytes_done{0}, chunks_done{0};
  std::mutex m;
  std::vector<Rec> log;
  size_t r = 0;
  uint64_t off = 0, chunk = 0;
};

static State* S = nullptr;
static std::vector<Rec> last_log;  // the stopped run's log, until the next start

static int finish(State& st, int i) {
  cudaError_t e = cudaEventSynchronize(st.e1[i]);
  if (e != cudaSuccess) return (int)e;
  float a = 0, b = 0;
  cudaEventElapsedTime(&a, st.start, st.e0[i]);
  cudaEventElapsedTime(&b, st.start, st.e1[i]);
  uint32_t root4 = 0;
  if (st.mode >= 0) {
    uint8_t root[64];
    bw::digest_bytes((const uint64_t*)(st.roots_h + 64 * i), root);
    root4 = (uint32_t)root[0] << 24 | (uint32_t)root[1] << 16 | (uint32_t)root[2] << 8 | root[3];
  }
  {
    std::lock_guard<std::mutex> g(st.m);
    st.log.push_back(Rec{a, b, st.slot_bytes[i], st.slot_region[i], root4, st.slot_chunk[i]});
  }
  st.bytes_done += st.slot_bytes[i];
  st.chunks_done += 1;
  st.busy[i] = 0;
  return 0;
}

// The next chunk's region and bytes, wrapping over the regions; blocks of 2,056 bytes for a receipt mode.
static void next_chunk(State& st, size_t& region, const uint8_t*& p, uint64_t& bytes) {
  const uint64_t unit = st.mode >= 0 ? bw::BLOCK_BYTES : 16;
  for (;;) {
    const Region& R = st.regions[st.r];
    const uint64_t usable = R.bytes / unit * unit;
    if (st.off + unit <= usable) {
      region = st.r;
      p = R.p + st.off;
      bytes = std::min<uint64_t>(st.chunk_bytes / unit * unit, usable - st.off);
      st.off += bytes;
      return;
    }
    st.r = (st.r + 1) % st.regions.size();
    st.off = 0;
  }
}

static int launch(State& st, int i) {
  size_t region;
  const uint8_t* p;
  uint64_t bytes;
  next_chunk(st, region, p, bytes);
  cudaError_t e = cudaEventRecord(st.e0[i], st.s);
  if (e != cudaSuccess) return (int)e;
  if (st.mode == MODE_READ) {
    const uint64_t n16 = bytes / 16, tile16 = st.tile_bytes / 16, tiles = (n16 + tile16 - 1) / tile16;
    const uint64_t grid = st.ctas > 0 && (uint64_t)st.ctas < tiles ? (uint64_t)st.ctas : tiles;
    tile_read<<<(unsigned)grid, 256, 0, st.s>>>((const uint4*)p, n16, (uint32_t)tile16, st.xr);
  } else {
    const uint64_t blocks = bytes / bw::BLOCK_BYTES;
    bw::Pass pass = bw::make_pass(st.mode == MODE_SK ? bw::TAG_B3 : st.mode, blocks, st.ws[i]);
    uint8_t seed[32];
    memcpy(seed, st.seed, 32);
    for (int k = 0; k < 8; ++k) seed[k] ^= (uint8_t)(st.chunk >> (8 * k));
    if (st.mode == MODE_SK) {
      sk_pass(pass, p, st.signs, seed, blocks, st.s);
    } else {
      bw::launch_tags(pass, p, seed, blocks, st.s);
      bw::launch_tree(pass, st.s);
    }
    e = cudaMemcpyAsync(st.roots_h + 64 * i, pass.nodes + pass.lv.off.back() * 8, 64, cudaMemcpyDeviceToHost, st.s);
    if (e != cudaSuccess) return (int)e;
  }
  e = cudaEventRecord(st.e1[i], st.s);
  if (e != cudaSuccess) return (int)e;
  st.slot_bytes[i] = bytes;
  st.slot_region[i] = (uint32_t)region;
  st.slot_chunk[i] = st.chunk++;
  st.busy[i] = 1;
  return (int)cudaGetLastError();
}

static void loop(State* sp) {
  State& st = *sp;
  cudaSetDevice(st.device);
  int i = 0, rc = 0;
  while (st.run.load() && !rc) {
    if (st.busy[i] && (rc = finish(st, i))) break;
    if (st.paused.load()) {
      bool any = false;
      for (int d = 0; d < st.depth && !rc; ++d) {
        const int j = (i + d) % st.depth;
        if (st.busy[j]) {
          rc = finish(st, j);
          any = true;
        }
      }
      if (!any) std::this_thread::sleep_for(std::chrono::microseconds(50));
      continue;
    }
    rc = launch(st, i);
    i = (i + 1) % st.depth;
  }
  for (int d = 0; d < st.depth && !rc; ++d)
    if (st.busy[(i + d) % st.depth]) rc = finish(st, (i + d) % st.depth);
  st.err = rc;
}

}  // namespace tu

extern "C" {

// Starts the top-up on `device` over `n` regions (device pointers and byte counts).  `priority` is the stream's
// (cudaDeviceGetStreamPriorityRange: 0 the least, negative greater).  `ctas` > 0 caps the read mode's grid (a receipt
// mode's grid follows its chunk: ro-blake3/v1 runs a CTA per 64 blocks, gf31-poly/v1 one per 256).  Returns 0 or a
// CUDA error.
int tu_start(int device, const uint64_t* ptrs, const uint64_t* bytes, int n, int mode, uint64_t chunk_bytes,
             uint64_t tile_bytes, int depth, int priority, const uint8_t* seed, int ctas) {
  if (tu::S || n <= 0 || depth <= 0 ||
      (mode != tu::MODE_READ && mode != bw::TAG_GF31 && mode != bw::TAG_B3 && mode != tu::MODE_SK))
    return -1;
  tu::last_log.clear();
  auto* st = new tu::State();
  st->device = device;
  st->mode = mode;
  st->depth = depth;
  st->priority = priority;
  st->ctas = ctas;
  st->chunk_bytes = chunk_bytes;
  st->tile_bytes = tile_bytes ? tile_bytes : (1ull << 16);
  memcpy(st->seed, seed, 32);
  for (int k = 0; k < n; ++k) st->regions.push_back(tu::Region{(const uint8_t*)ptrs[k], bytes[k]});
  CK(cudaSetDevice(device));
  CK(cudaStreamCreateWithPriority(&st->s, cudaStreamNonBlocking, priority));
  CK(cudaEventCreate(&st->start));
  st->e0.resize(depth);
  st->e1.resize(depth);
  st->ws.assign(depth, nullptr);
  st->slot_bytes.assign(depth, 0);
  st->slot_chunk.assign(depth, 0);
  st->slot_region.assign(depth, 0);
  st->busy.assign(depth, 0);
  for (int i = 0; i < depth; ++i) {
    CK(cudaEventCreate(&st->e0[i]));
    CK(cudaEventCreateWithFlags(&st->e1[i], cudaEventBlockingSync));
    if (mode >= 0)
      CK(cudaMalloc(&st->ws[i], bw::workspace_bytes(mode == tu::MODE_SK ? bw::TAG_B3 : mode, chunk_bytes / bw::BLOCK_BYTES)));
  }
  if (mode == tu::MODE_SK) {
    std::vector<uint32_t> sg(tu::SK_WORDS * tu::SK_ROWS);
    tu::sk_signs(seed, sg.data());
    CK(cudaMalloc(&st->signs, sg.size() * 4));
    CK(cudaMemcpy(st->signs, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute(tu::sk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tu::SK_SMEM));
  }
  CK(cudaMallocHost(&st->roots_h, 64 * depth));
  CK(cudaMalloc(&st->xr, 4));
  CK(cudaMemsetAsync(st->xr, 0, 4, st->s));
  CK(cudaEventRecord(st->start, st->s));
  CK(cudaStreamSynchronize(st->s));
  st->log.reserve(1 << 20);
  st->run = 1;
  tu::S = st;
  st->th = std::thread(tu::loop, st);
  return 0;
}

void tu_pause(int on) {
  if (tu::S) tu::S->paused = on;
}

// The start event's time on the GPU's clock, as an event recorded now on `stream` would read: lets a caller place its
// own events on the top-up's timeline.  Returns the elapsed milliseconds of an event recorded on `stream` after the
// stream's prior work, or -1.
double tu_now_ms(void* stream) {
  if (!tu::S) return -1;
  cudaEvent_t e;
  if (cudaEventCreate(&e) != cudaSuccess) return -1;
  cudaEventRecord(e, (cudaStream_t)stream);
  cudaEventSynchronize(e);
  float ms = -1;
  cudaEventElapsedTime(&ms, tu::S->start, e);
  cudaEventDestroy(e);
  return ms;
}

// The start event, for callers that record their own events and read them against it.
void* tu_start_event() { return tu::S ? (void*)tu::S->start : nullptr; }

// Bytes and chunks finished so far.
void tu_progress(uint64_t* out) {
  out[0] = tu::S ? tu::S->bytes_done.load() : 0;
  out[1] = tu::S ? tu::S->chunks_done.load() : 0;
}

// Copies up to `max` log records from `first` into `out` (6 doubles each: t0_ms, t1_ms, bytes, region, root4, chunk),
// the running top-up's or, once stopped, its last run's.
int tu_log(uint64_t first, double* out, int max) {
  std::unique_lock<std::mutex> g;
  if (tu::S) g = std::unique_lock<std::mutex>(tu::S->m);
  const std::vector<tu::Rec>& log = tu::S ? tu::S->log : tu::last_log;
  int n = 0;
  for (uint64_t k = first; k < log.size() && n < max; ++k, ++n) {
    const tu::Rec& r = log[k];
    double* o = out + 6 * n;
    o[0] = r.t0_ms;
    o[1] = r.t1_ms;
    o[2] = (double)r.bytes;
    o[3] = r.region;
    o[4] = r.root4;
    o[5] = (double)r.chunk;
  }
  return n;
}

// Stops the thread after its in-flight chunks; returns its error (0 if none) and frees everything.
int tu_stop() {
  tu::State* st = tu::S;
  if (!st) return -1;
  st->run = 0;
  st->th.join();
  int rc = st->err.load();
  for (int i = 0; i < st->depth; ++i) {
    cudaEventDestroy(st->e0[i]);
    cudaEventDestroy(st->e1[i]);
    if (st->ws[i]) cudaFree(st->ws[i]);
  }
  cudaEventDestroy(st->start);
  cudaFreeHost(st->roots_h);
  cudaFree(st->xr);
  if (st->signs) cudaFree(st->signs);
  cudaStreamDestroy(st->s);
  tu::last_log = std::move(st->log);
  delete st;
  tu::S = nullptr;
  return rc;
}

// One chunk's receipt root, computed synchronously the way the thread computes it (the gate): mode 0 or 2 over `blocks`
// 2,056-byte blocks at `store` under `seed`; the 64-byte root into `root`.
int tu_root(int mode, const void* store, uint64_t blocks, const uint8_t* seed, uint8_t* root) {
  void* ws = nullptr;
  CK(cudaMalloc(&ws, bw::workspace_bytes(mode, blocks)));
  bw::Pass pass = bw::make_pass(mode, blocks, ws);
  bw::launch_tags(pass, store, seed, blocks, 0);
  bw::launch_tree(pass, 0);
  int rc = bw::read_root(pass, root, 0);
  cudaFree(ws);
  return rc;
}

// sk16's gate: the device's digests of `blocks` blocks at `store` (device memory) under `seed` against the host's, and
// the pass's root into `root`.  Returns the number of blocks whose digests differ, or -1 on a CUDA error.
int tu_sk_gate(const void* store, uint64_t blocks, const uint8_t* seed, uint8_t* root) {
  std::vector<uint32_t> sg(tu::SK_WORDS * tu::SK_ROWS);
  tu::sk_signs(seed, sg.data());
  uint32_t* signs = nullptr;
  void* ws = nullptr;
  if (cudaMalloc(&signs, sg.size() * 4) != cudaSuccess) return -1;
  if (cudaMalloc(&ws, bw::workspace_bytes(bw::TAG_B3, blocks)) != cudaSuccess) return -1;
  cudaMemcpy(signs, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(tu::sk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tu::SK_SMEM);
  bw::Pass pass = bw::make_pass(bw::TAG_B3, blocks, ws);
  tu::sk_pass(pass, store, signs, seed, blocks, 0);
  if (bw::read_root(pass, root, 0)) return -1;
  std::vector<uint32_t> dev(blocks * 8);
  std::vector<uint8_t> E(blocks * bw::BLOCK_BYTES);
  if (cudaMemcpy(dev.data(), pass.tags, dev.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
  if (cudaMemcpy(E.data(), store, E.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
  const bw::Key3 key = bw::b3_key(seed);
  int bad = 0;
  for (uint64_t j = 0; j < blocks; ++j) {
    uint32_t y[8];
    tu::sk_host(E.data() + j * bw::BLOCK_BYTES, j, sg.data(), key, y);
    if (memcmp(y, dev.data() + j * 8, 32)) ++bad;
  }
  cudaFree(ws);
  cudaFree(signs);
  return bad;
}

// One chunk's sk16 root, computed synchronously the way the thread computes it.
int tu_sk_root(const void* store, uint64_t blocks, const uint8_t* seed, uint8_t* root) {
  std::vector<uint32_t> sg(tu::SK_WORDS * tu::SK_ROWS);
  tu::sk_signs(seed, sg.data());
  uint32_t* signs = nullptr;
  void* ws = nullptr;
  CK(cudaMalloc(&signs, sg.size() * 4));
  CK(cudaMalloc(&ws, bw::workspace_bytes(bw::TAG_B3, blocks)));
  CK(cudaMemcpy(signs, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(tu::sk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tu::SK_SMEM));
  bw::Pass pass = bw::make_pass(bw::TAG_B3, blocks, ws);
  tu::sk_pass(pass, store, signs, seed, blocks, 0);
  int rc = bw::read_root(pass, root, 0);
  cudaFree(ws);
  cudaFree(signs);
  return rc;
}

// A plain read of `bytes` at `p` by `grid` CTAs of `tile_bytes` tiles (grid 0: one CTA a tile), for R.
int tu_read(const void* p, uint64_t bytes, uint64_t tile_bytes, void* stream, uint32_t* xr) {
  const uint64_t n16 = bytes / 16, tile16 = tile_bytes / 16;
  tu::tile_read<<<(unsigned)((n16 + tile16 - 1) / tile16), 256, 0, (cudaStream_t)stream>>>((const uint4*)p, n16,
                                                                                         (uint32_t)tile16, xr);
  return (int)cudaGetLastError();
}

}  // extern "C"
