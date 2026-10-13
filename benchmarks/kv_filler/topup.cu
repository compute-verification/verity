// The filler top-up: a host thread of its own reads registered regions of HBM on a low-priority stream, chunk after
// chunk, each chunk with its receipts, so that a window's reads approach R·T while a serving process runs beside it and
// its Python never waits on the top-up.
//
//   mode -1  read only: every 16 bytes loaded once and XOR-folded (the floor of a receipt's cost)
//   mode  0  gf31-poly/v1 over 2,056-byte blocks, the bandwidth audit's uncertified fast check
//   mode  2  ro-blake3/v1 over 2,056-byte blocks, the bandwidth audit's certified receipt, with its leaves and tree
//   mode  3  sk16: the cost of a sketch-then-hash receipt over the same blocks (ra-sketch-tag's shape, not its format):
//            each block's 2,056 bytes as int8 against 16 rows of ±1 signs (dp4a), the 16 int32 sums hashed by one keyed
//            BLAKE3 compression under the block's index, then ro-blake3/v1's leaves and tree over those 32-byte digests;
//            the signs are drawn once a run from its seed (a deployment draws them a pass, F29: the cost is the same)
//   mode  4  fill-sk16/v0 (`fill.cu`): 4,096-byte blocks sketched on the tensor cores against 16 rows of ±1 signs, each
//            block's sums hashed by one keyed BLAKE3 compression and each 32 blocks' leaves into a unit root; a chunk is
//            whole units, read by a fixed grid (`ctas`, default one CTA an SM) of 4 warps a CTA, a unit a warp at a time,
//            the chunk's check word the XOR of its units' first root words. Chunks alternate between two streams so that
//            one chunk's tail overlaps the next one's start. Signs drawn once a run, as sk16's
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
#include "readers.cu"
#include "fill.cu"

#include <atomic>
#include <mutex>
#include <thread>

namespace tu {

constexpr int MODE_READ = -1, MODE_SK = 3, MODE_FILL = 4;
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

// `count` pages of page16 16-byte words out of n_pages, CTA b taking the j-th for j = b, b + gridDim.x, ...: page
// (first + j) mod n_pages in order (`random` = 0, a weight stream), else a SplitMix64 hash of (seed, j) mod n_pages
// (paged KV gathers).
__global__ void __launch_bounds__(256) page_read(const uint4* __restrict__ p, uint64_t n_pages, uint32_t page16,
                                                 uint64_t count, uint64_t first, int random, uint64_t seed,
                                                 uint32_t* out) {
  uint32_t acc = 0;
  for (uint64_t j = blockIdx.x; j < count; j += gridDim.x) {
    uint64_t z = seed + (first + j) * 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    const uint4* q = p + (random ? z % n_pages : (first + j) % n_pages) * page16;
    for (uint32_t i = threadIdx.x; i < page16; i += 256) {
      uint4 v = __ldcs(q + i);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, d);
  if ((threadIdx.x & 31) == 0 && acc == 0x9e3779b9u) atomicXor(out, acc);
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

// fill-sk16/v0's signs: SHA-512(seed ‖ "verity/kv-filler/fill-sk16/v0/signs" ‖ u32le i) for i = 0..127, 8 KiB.
static void fill_signs(const uint8_t seed[32], uint32_t* out) {
  static const char name[] = "verity/kv-filler/fill-sk16/v0/signs";
  std::vector<uint8_t> msg(seed, seed + 32);
  msg.insert(msg.end(), name, name + sizeof(name) - 1);
  msg.resize(msg.size() + 4);
  uint8_t* bytes = (uint8_t*)out;
  for (uint32_t i = 0; i < 128; ++i) {
    for (int k = 0; k < 4; ++k) msg[msg.size() - 4 + k] = (uint8_t)(i >> (8 * k));
    uint64_t d[8];
    bw::sha512_host(msg.data(), msg.size(), d);
    bw::digest_bytes(d, bytes + 64 * i);
  }
}

static fl::Key fill_key(const uint8_t seed[32]) {
  fl::Key k;
  memcpy(k.k, seed, 32);
  return k;
}

// The plan over `regions`: each region's whole units, in order.
// The kernel's loads are 16-byte vectors from each region's base.
static bool fill_ok(const std::vector<Region>& regions) {
  if (regions.empty() || regions.size() > (size_t)fl::Regions::MAX) return false;
  for (const Region& r : regions)
    if ((uintptr_t)r.p % 16) return false;
  return true;
}

static fl::Regions fill_regions(const std::vector<Region>& regions) {
  fl::Regions R{};
  R.n = (int)regions.size();
  R.first_unit[0] = 0;
  for (int i = 0; i < R.n; ++i) {
    R.base[i] = regions[i].p;
    R.first_unit[i + 1] = R.first_unit[i] + regions[i].bytes / fl::UNIT_BYTES;
  }
  return R;
}

// Warps a fill CTA (4, 8 or 16; tu_fill_warps): at 16 one CTA holds a whole SM's registers.
static std::atomic<int> fill_warps{4};

// The fill kernels' gate (fill.cu), one for the process, and the stream the host's own gate writes go on.
static fl::Gate* gate = nullptr;
static cudaStream_t gate_stream = nullptr;
static std::mutex gate_mu;

static int gate_init() {
  std::lock_guard<std::mutex> lk(gate_mu);
  if (gate) return 0;
  int lo = 0, hi = 0;
  cudaDeviceGetStreamPriorityRange(&lo, &hi);
  cudaError_t e = cudaStreamCreateWithPriority(&gate_stream, cudaStreamNonBlocking, hi);
  if (e == cudaSuccess) e = cudaMalloc(&gate, sizeof(fl::Gate));
  if (e != cudaSuccess) return (int)e;
  fl::gate_set<<<1, 1, 0, gate_stream>>>(gate, fl::ALL, fl::ALL, 1);
  return (int)cudaStreamSynchronize(gate_stream);
}

static uint32_t gate_value(int ctas) { return ctas < 0 ? fl::ALL : (uint32_t)ctas; }

static std::atomic<int> gate_closed{0};  // both ways: the thread launches no chunk

static int gate_set(int busy_ctas, int idle_ctas, bool reset) {
  if (int rc = gate_init()) return rc;
  gate_closed = busy_ctas == 0 && idle_ctas == 0;
  fl::gate_set<<<1, 1, 0, gate_stream>>>(gate, gate_value(busy_ctas), gate_value(idle_ctas), reset);
  return (int)cudaStreamSynchronize(gate_stream);
}

// One fill chunk: units [unit0, unit0 + units) of the plan, roots into ws, the check word into ws's tail.
static int fill_launch(const fl::Regions& R, uint64_t unit0, uint64_t units, const uint32_t* signs,
                       const uint8_t seed[32], void* ws, int ctas, cudaStream_t st) {
  uint32_t* roots = (uint32_t*)ws;
  auto* ctr = (unsigned long long*)(roots + 8 * units);
  uint32_t* xr = (uint32_t*)(ctr + 1);
  cudaMemsetAsync(ctr, 0, 16, st);
  int grid = ctas;
  if (grid <= 0) {
    int d = 0;
    cudaGetDevice(&d);
    cudaDeviceGetAttribute(&grid, cudaDevAttrMultiProcessorCount, d);
  }
  if (int rc = gate_init()) return rc;
  const fl::Key K = fill_key(seed);
  switch (fill_warps.load()) {
    case 4: fl::fill_kernel<4><<<grid, 128, 0, st>>>(R, unit0, units, signs, K, roots, ctr, xr, gate); break;
    case 8: fl::fill_kernel<8><<<grid, 256, 0, st>>>(R, unit0, units, signs, K, roots, ctr, xr, gate); break;
    case 16: fl::fill_kernel<16><<<grid, 512, 0, st>>>(R, unit0, units, signs, K, roots, ctr, xr, gate); break;
    default: return -1;
  }
  return (int)cudaGetLastError();
}

static size_t fill_ws_bytes(uint64_t units) { return 32 * units + 16; }

struct State {
  int device = 0, mode = MODE_READ, depth = 2, priority = 0;
  std::atomic<int> ctas{0};
  uint64_t chunk_bytes = 1ull << 28, tile_bytes = 1ull << 16;
  std::vector<Region> regions;
  uint8_t seed[32];
  cudaStream_t s = nullptr;
  cudaEvent_t start = nullptr;
  fl::Regions plan{};
  uint64_t plan_units = 0, fill_unit = 0;
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
  if (st.mode == MODE_FILL) {
    uint64_t taken = 0;
    memcpy(&taken, st.roots_h + 64 * i, 8);
    memcpy(&root4, st.roots_h + 64 * i + 8, 4);
    st.slot_bytes[i] = std::min<uint64_t>(taken * fl::UNIT_BYTES, st.slot_bytes[i]);
  } else if (st.mode >= 0) {
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
// Under a CTA cap a receipt mode's chunk is at most the cap's CTAs of blocks (ro-blake3/v1 64 a CTA, the others 256).
static void next_chunk(State& st, size_t& region, const uint8_t*& p, uint64_t& bytes) {
  const uint64_t unit = st.mode >= 0 ? bw::BLOCK_BYTES : 16;
  uint64_t limit = st.chunk_bytes / unit * unit;
  const int cap = st.ctas.load();
  if (cap > 0 && st.mode >= 0) limit = std::min<uint64_t>(limit, (uint64_t)cap * (st.mode == bw::TAG_B3 ? 64 : 256) * unit);
  for (;;) {
    const Region& R = st.regions[st.r];
    const uint64_t usable = R.bytes / unit * unit;
    if (st.off + unit <= usable) {
      region = st.r;
      p = R.p + st.off;
      bytes = std::min<uint64_t>(limit, usable - st.off);
      st.off += bytes;
      return;
    }
    st.r = (st.r + 1) % st.regions.size();
    st.off = 0;
  }
}

static int launch_fill(State& st, int i) {
  cudaStream_t s = st.s;
  const uint64_t units = std::min<uint64_t>(st.chunk_bytes / fl::UNIT_BYTES, st.plan_units - st.fill_unit);
  const uint64_t unit0 = st.fill_unit;
  st.fill_unit = (st.fill_unit + units) % st.plan_units;
  uint8_t seed[32];
  memcpy(seed, st.seed, 32);
  for (int k = 0; k < 8; ++k) seed[k] ^= (uint8_t)(st.chunk >> (8 * k));
  cudaError_t e = cudaEventRecord(st.e0[i], s);
  if (e != cudaSuccess) return (int)e;
  int rc = fill_launch(st.plan, unit0, units, st.signs, seed, st.ws[i], st.ctas.load(), s);
  if (rc) return rc;
  e = cudaMemcpyAsync(st.roots_h + 64 * i, (uint8_t*)st.ws[i] + 32 * units, 12, cudaMemcpyDeviceToHost, s);
  if (e != cudaSuccess) return (int)e;
  e = cudaEventRecord(st.e1[i], s);
  if (e != cudaSuccess) return (int)e;
  st.slot_bytes[i] = units * fl::UNIT_BYTES;
  st.slot_region[i] = (uint32_t)unit0;
  st.slot_chunk[i] = st.chunk++;
  st.busy[i] = 1;
  return (int)cudaGetLastError();
}

static int launch(State& st, int i) {
  if (st.mode == MODE_FILL) return launch_fill(st, i);
  size_t region;
  const uint8_t* p;
  uint64_t bytes;
  next_chunk(st, region, p, bytes);
  cudaError_t e = cudaEventRecord(st.e0[i], st.s);
  if (e != cudaSuccess) return (int)e;
  if (st.mode == MODE_READ) {
    const uint64_t n16 = bytes / 16, tile16 = st.tile_bytes / 16, tiles = (n16 + tile16 - 1) / tile16;
    const int cap = st.ctas.load();
    const uint64_t grid = cap > 0 && (uint64_t)cap < tiles ? (uint64_t)cap : tiles;
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
    if (st.paused.load() || (st.mode == MODE_FILL && gate_closed.load())) {
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

// In fill mode tu_pause and tu_throttle move the gate at once (paused: no CTA; a cap of 0: every CTA) instead of the
// next chunk's launch.
static std::atomic<int> host_paused{0}, host_cap{0};

static void host_gate() {
  const int c = host_paused ? 0 : host_cap ? host_cap.load() : -1;
  gate_set(c, c, false);
}

}  // namespace tu

extern "C" {

// Starts the top-up on `device` over `n` regions (device pointers and byte counts).  `priority` is the stream's
// (cudaDeviceGetStreamPriorityRange: 0 the least, negative greater).  `ctas` > 0 caps the grid: the read mode's CTAs, or
// a receipt mode's chunk at that many CTAs of blocks.  Returns 0 or a CUDA error.
int tu_start(int device, const uint64_t* ptrs, const uint64_t* bytes, int n, int mode, uint64_t chunk_bytes,
             uint64_t tile_bytes, int depth, int priority, const uint8_t* seed, int ctas) {
  if (tu::S || n <= 0 || depth <= 0 ||
      (mode != tu::MODE_READ && mode != bw::TAG_GF31 && mode != bw::TAG_B3 && mode != tu::MODE_SK &&
       mode != tu::MODE_FILL))
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
  if (mode == tu::MODE_FILL) {
    if (!tu::fill_ok(st->regions)) return -1;
    st->plan = tu::fill_regions(st->regions);
    st->plan_units = st->plan.first_unit[st->plan.n];
    if (st->plan_units == 0) return -1;
    tu::host_paused = 0;
    tu::host_cap = 0;
    if (int rc = tu::gate_set(-1, -1, true)) return rc;
  }
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
    if (mode == tu::MODE_FILL)
      CK(cudaMalloc(&st->ws[i], tu::fill_ws_bytes(chunk_bytes / fl::UNIT_BYTES)));
    else if (mode >= 0)
      CK(cudaMalloc(&st->ws[i], bw::workspace_bytes(mode == tu::MODE_SK ? bw::TAG_B3 : mode, chunk_bytes / bw::BLOCK_BYTES)));
  }
  if (mode == tu::MODE_SK) {
    std::vector<uint32_t> sg(tu::SK_WORDS * tu::SK_ROWS);
    tu::sk_signs(seed, sg.data());
    CK(cudaMalloc(&st->signs, sg.size() * 4));
    CK(cudaMemcpy(st->signs, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute(tu::sk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tu::SK_SMEM));
  }
  if (mode == tu::MODE_FILL) {
    std::vector<uint32_t> sg(fl::SIGN_WORDS);
    tu::fill_signs(seed, sg.data());
    CK(cudaMalloc(&st->signs, sg.size() * 4));
    CK(cudaMemcpy(st->signs, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice));
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
  if (!tu::S) return;
  if (tu::S->mode == tu::MODE_FILL) {
    tu::host_paused = on;
    tu::host_gate();
  } else {
    tu::S->paused = on;
  }
}

// Sets the CTA cap from the next chunk on (0: none).
void tu_throttle(int ctas) {
  if (!tu::S) return;
  if (tu::S->mode == tu::MODE_FILL) {
    tu::host_cap = ctas;
    tu::host_gate();
  } else {
    tu::S->ctas = ctas;
  }
}

// fill-sk16/v0's gate, now (fill.cu): `busy_ctas` CTAs take units while a step is on the GPU, `idle_ctas` otherwise
// (negative: every CTA; 0: none). A chunk under a closed gate waits; tu_stop opens it.
int tu_gate(int busy_ctas, int idle_ctas) { return tu::gate_set(busy_ctas, idle_ctas, false); }

// A step's start (+1) or end (-1) on `stream`, ordered with the step's own kernels, so the filler's grid moves at the
// step's edges on the GPU's clock. tu_gate_reset() zeroes the count and opens the gate.
int tu_gate_step(int delta, void* stream) {
  if (int rc = tu::gate_init()) return rc;
  fl::gate_step<<<1, 1, 0, (cudaStream_t)stream>>>(tu::gate, delta);
  return (int)cudaGetLastError();
}

int tu_gate_reset() { return tu::gate_set(-1, -1, true); }

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
  if (st->mode == tu::MODE_FILL) tu::gate_set(-1, -1, true);
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

// One chunk's sk16 root, computed synchronously the way the thread computes it: the chunk's `seed` keys the hash and
// names the leaves, the run's `sign_seed` gives the signs (the thread draws them once, at its start).
int tu_sk_root(const void* store, uint64_t blocks, const uint8_t* seed, const uint8_t* sign_seed, uint8_t* root) {
  std::vector<uint32_t> sg(tu::SK_WORDS * tu::SK_ROWS);
  tu::sk_signs(sign_seed, sg.data());
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

int tu_page_read(const void* p, uint64_t bytes, uint64_t page_bytes, uint64_t count, uint64_t first, int random,
                 uint64_t seed, int ctas, void* stream, uint32_t* xr) {
  tu::page_read<<<ctas, 256, 0, (cudaStream_t)stream>>>((const uint4*)p, bytes / page_bytes,
                                                        (uint32_t)(page_bytes / 16), count, first, random, seed, xr);
  return (int)cudaGetLastError();
}

// fill-sk16/v0's gate: units [unit0, unit0 + units) of the plan over `n` regions under the chunk `seed` (key) and
// `sign_seed` (signs), computed as the thread computes a chunk, with `ctas` CTAs; the units' roots (32 bytes each) and
// the check word copied to host memory `roots` (32 · units + 4 bytes).
int tu_fill_units(const uint64_t* ptrs, const uint64_t* bytes, int n, uint64_t unit0, uint64_t units,
                  const uint8_t* seed, const uint8_t* sign_seed, int ctas, uint8_t* roots) {
  if (n <= 0 || n > fl::Regions::MAX) return -1;
  std::vector<tu::Region> regions;
  for (int k = 0; k < n; ++k) regions.push_back(tu::Region{(const uint8_t*)ptrs[k], bytes[k]});
  if (!tu::fill_ok(regions)) return -1;
  const fl::Regions R = tu::fill_regions(regions);
  std::vector<uint32_t> sg(fl::SIGN_WORDS);
  tu::fill_signs(sign_seed, sg.data());
  uint32_t* signs = nullptr;
  void* ws = nullptr;
  CK(cudaMalloc(&signs, sg.size() * 4));
  CK(cudaMalloc(&ws, tu::fill_ws_bytes(units)));
  CK(cudaMemcpy(signs, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice));
  int rc = tu::fill_launch(R, unit0, units, signs, seed, ws, ctas, 0);
  if (!rc) rc = (int)cudaDeviceSynchronize();
  if (!rc) rc = (int)cudaMemcpy(roots, ws, 32 * units, cudaMemcpyDeviceToHost);
  if (!rc) rc = (int)cudaMemcpy(roots + 32 * units, (uint8_t*)ws + 32 * units + 8, 4, cudaMemcpyDeviceToHost);
  cudaFree(ws);
  cudaFree(signs);
  return rc;
}

int tu_fill_warps(int w) {
  if (w != 4 && w != 8 && w != 16) return -1;
  tu::fill_warps = w;
  return 0;
}

// The 8 KiB of fill-sk16/v0 signs from `seed`, for the reference to check against.
void tu_fill_signs(const uint8_t* seed, uint8_t* out) { tu::fill_signs(seed, (uint32_t*)out); }

}  // extern "C"

// --- fill-sk16/v2: the filler by sequence number (fill.cu's fill_seq_kernel) ----------------------------------------
// One thread keeps BASE kernels of `base_ctas` CTAs queued (two deep, `base_budget` units each) and, with idle on, one
// IDLE kernel of every SM's CTA at a time, launched again at once after one that took units and after `IDLE_BACKOFF`
// after one that took none; sq_step_begin/sq_step_end bracket a step on its own stream (the gate's busy count) and
// launch a STEP kernel of `ctas` CTAs beside it. Each epoch of 2^shift units is logged once all its units are done
// (t0 the previous epoch's end, t1 its own, on the start event's clock), and the last, partial one at sq_stop.
namespace sq {

constexpr uint32_t EPOCHS = 4096;
constexpr uint64_t IN_FLIGHT = 1024;  // more than the CTAs of every role at once

static uint64_t limit(uint32_t shift, uint64_t next_epoch) { return ((next_epoch + EPOCHS) << shift) - IN_FLIGHT; }
constexpr auto IDLE_BACKOFF = std::chrono::microseconds(200);

struct State {
  int device = 0, warps = 4, base_ctas = 0, sms = 0;
  std::atomic<int> prefill_stages{0};
  std::atomic<int> step_warps{0}, depth{fl::DEPTH};  // STEP's and PREFILL's warps a CTA (0: `warps`), loads in flight
  std::atomic<int> probe{0};  // fill.cu's PROBE for every launch from now (sq_probe; the receipts are then wrong)  // the PREFILL role's TMA stages (fill.cu's fill_seq_bulk_kernel), 0: 16-byte loads
  uint32_t shift = 8;
  uint64_t base_budget = 4096, idle_budget = 1ull << 22, step_budget = 1ull << 22;
  std::atomic<int> idle_on{0};
  fl::Regions plan{};
  uint8_t seed[32];
  fl::Signs sg{};
  fl::Seq* seq = nullptr;
  fl::Ring ring{};
  unsigned long long* taken = nullptr;  // [0, 1] the base slots', [2] idle's, [3] step's, [4] prefill's
  unsigned long long* taken_h = nullptr;  // pinned: idle's after it ends
  unsigned long long* g0 = nullptr;  // the globaltimer at the start event
  unsigned long long g0_h = 0;
  uint64_t lim = 0;  // the host's copy of seq->limit, the source of its async copy
  cudaStream_t s_base = nullptr, s_idle = nullptr, s_step = nullptr, s_pre = nullptr, s_h = nullptr;
  cudaEvent_t start = nullptr, base_done[2] = {nullptr, nullptr}, idle_done = nullptr, step_ev = nullptr, step_done = nullptr,
              pre_done = nullptr;
  std::mutex step_mu;
  std::thread th;
  std::atomic<int> run{0}, err{0};
  std::atomic<uint64_t> units_done{0}, epochs_done{0}, steps{0}, step_launches{0}, prefill_steps{0},
      prefill_launches{0};
  std::mutex m;
  std::vector<tu::Rec> log;
};

static State* S = nullptr;

static int launch(State& st, int role, int grid, unsigned long long* taken, uint64_t budget, cudaStream_t s) {
  cudaError_t e = cudaMemsetAsync(taken, 0, 8, s);
  if (e != cudaSuccess) return (int)e;
#define SQ_B(S) \
  fl::fill_seq_bulk_kernel<S><<<grid, 32, S * fl::TB_STAGE, s>>>(st.plan, st.sg, tu::gate, st.seq, st.ring, role, taken, budget)
  const int sw = st.step_warps.load();
  const int warps = (role == fl::STEP || role == fl::PREFILL) && sw ? sw : st.warps, depth = st.depth.load();
  if (const int pr = st.probe.load()) {  // the probes run at 4 or 16 warps and the default depth, or 3 stages
    const int stages = role == fl::PREFILL ? st.prefill_stages.load() : 0;
#define SQ_P(PR)                                                                                                       \
  if (stages)                                                                                                         \
    fl::fill_seq_bulk_kernel<3, PR><<<grid, 32, 3 * fl::TB_STAGE, s>>>(st.plan, st.sg, tu::gate, st.seq, st.ring, role, \
                                                                       taken, budget);                                \
  else if (warps == 16)                                                                                               \
    fl::fill_seq_kernel<16, PR><<<grid, 512, 0, s>>>(st.plan, st.sg, tu::gate, st.seq, st.ring, role, taken, budget); \
  else                                                                                                                \
    fl::fill_seq_kernel<4, PR><<<grid, 128, 0, s>>>(st.plan, st.sg, tu::gate, st.seq, st.ring, role, taken, budget)
    if ((stages && stages != 3) || (!stages && ((warps != 4 && warps != 16) || depth != fl::DEPTH))) return -1;
    switch (pr) {
      case 1: SQ_P(1); break;
      case 2: SQ_P(2); break;
      case 3: SQ_P(3); break;
      default: return -1;
    }
#undef SQ_P
    return (int)cudaGetLastError();
  }
  if (const int stages = role == fl::PREFILL ? st.prefill_stages.load() : 0) {
    switch (stages) {
      case 2: SQ_B(2); break;
      case 3: SQ_B(3); break;
      case 4: SQ_B(4); break;
      default: return -1;
    }
    return (int)cudaGetLastError();
  }
#undef SQ_B
#define SQ_L(W, D) \
  fl::fill_seq_kernel<W, 0, D><<<grid, W * 32, 0, s>>>(st.plan, st.sg, tu::gate, st.seq, st.ring, role, taken, budget)
  switch (100 * warps + depth) {
    case 402: SQ_L(4, 2); break;
    case 408: SQ_L(4, 8); break;
    case 802: SQ_L(8, 2); break;
    case 804: SQ_L(8, 4); break;
    case 808: SQ_L(8, 8); break;
    case 1602: SQ_L(16, 2); break;
    default: return -1;
  }
#undef SQ_L
  return (int)cudaGetLastError();
}

// Logs every epoch from `next` on whose units are all done, each over its first and last unit's completion; with `last`
// (every kernel ended, `total` units taken), the partial epoch after them too.
static int harvest(State& st, uint64_t& next, std::vector<uint32_t>& cnt, std::vector<uint32_t>& xr,
                   std::vector<unsigned long long>& t0, std::vector<unsigned long long>& t1, bool last, uint64_t total) {
  cudaError_t e = cudaMemcpyAsync(cnt.data(), st.ring.cnt, 4 * EPOCHS, cudaMemcpyDeviceToHost, st.s_h);
  if (e == cudaSuccess) e = cudaMemcpyAsync(xr.data(), st.ring.xr, 4 * EPOCHS, cudaMemcpyDeviceToHost, st.s_h);
  if (e == cudaSuccess) e = cudaMemcpyAsync(t0.data(), st.ring.t0, 8 * EPOCHS, cudaMemcpyDeviceToHost, st.s_h);
  if (e == cudaSuccess) e = cudaMemcpyAsync(t1.data(), st.ring.t1, 8 * EPOCHS, cudaMemcpyDeviceToHost, st.s_h);
  if (e == cudaSuccess) e = cudaStreamSynchronize(st.s_h);
  if (e != cudaSuccess) return (int)e;
  const uint32_t full = 1u << st.shift;
  const uint64_t next0 = next;
  for (;;) {
    const uint32_t slot = (uint32_t)(next & (EPOCHS - 1));
    const uint64_t first = next << st.shift;
    uint32_t n = cnt[slot];
    if (n != full) {
      if (!last || total <= first) break;
      if (n != total - first) return -2;  // every unit taken is done once every kernel has ended
    }
    {
      std::lock_guard<std::mutex> g(st.m);
      st.log.push_back(tu::Rec{(double)(t0[slot] - st.g0_h) / 1e6, (double)(t1[slot] - st.g0_h) / 1e6,
                               (uint64_t)n * fl::UNIT_BYTES, slot, xr[slot], next});
    }
    st.units_done += n;
    st.epochs_done += 1;
    ++next;
    if (n != full || next - next0 == EPOCHS) break;
  }
  // the logged slots cleared as at most two runs, before the limit lets them be taken again
  for (uint64_t k = next0; k < next && e == cudaSuccess;) {
    const uint32_t a = (uint32_t)(k & (EPOCHS - 1)), m = (uint32_t)std::min<uint64_t>(next - k, EPOCHS - a);
    e = cudaMemsetAsync(st.ring.cnt + a, 0, 4ull * m, st.s_h);
    if (e == cudaSuccess) e = cudaMemsetAsync(st.ring.xr + a, 0, 4ull * m, st.s_h);
    if (e == cudaSuccess) e = cudaMemsetAsync(st.ring.t0 + a, 0xFF, 8ull * m, st.s_h);
    if (e == cudaSuccess) e = cudaMemsetAsync(st.ring.t1 + a, 0, 8ull * m, st.s_h);
    k += m;
  }
  if (e == cudaSuccess && next != next0) {
    st.lim = limit(st.shift, next);
    e = cudaMemcpyAsync(&st.seq->limit, &st.lim, 8, cudaMemcpyHostToDevice, st.s_h);
  }
  return (int)e;
}

static void loop(State* sp) {
  State& st = *sp;
  cudaSetDevice(st.device);
  std::vector<uint32_t> cnt(EPOCHS), xr(EPOCHS);
  std::vector<unsigned long long> t0(EPOCHS), t1(EPOCHS);
  uint64_t next = 0;
  bool base_busy[2] = {false, false}, idle_busy = false;
  auto idle_at = std::chrono::steady_clock::now();
  int rc = 0;
  while (st.run.load() && !rc) {
    const bool closed = tu::gate_closed.load();
    for (int i = 0; i < 2 && !rc; ++i) {
      if (base_busy[i] && cudaEventQuery(st.base_done[i]) == cudaSuccess) base_busy[i] = false;
      if (!base_busy[i] && st.base_ctas > 0 && !closed) {
        rc = launch(st, fl::BASE, st.base_ctas, st.taken + i, st.base_budget, st.s_base);
        if (!rc) rc = (int)cudaEventRecord(st.base_done[i], st.s_base);
        base_busy[i] = true;
      }
    }
    const auto now = std::chrono::steady_clock::now();
    if (idle_busy && cudaEventQuery(st.idle_done) == cudaSuccess) {
      idle_busy = false;
      idle_at = *st.taken_h ? now : now + IDLE_BACKOFF;
    }
    if (!rc && !idle_busy && st.idle_on.load() && !closed && now >= idle_at) {
      rc = launch(st, fl::IDLE, st.sms, st.taken + 2, st.idle_budget, st.s_idle);
      if (!rc) rc = (int)cudaMemcpyAsync(st.taken_h, st.taken + 2, 8, cudaMemcpyDeviceToHost, st.s_idle);
      if (!rc) rc = (int)cudaEventRecord(st.idle_done, st.s_idle);
      idle_busy = true;
    }
    if (!rc) rc = harvest(st, next, cnt, xr, t0, t1, false, 0);
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  if (!rc) {
    fl::Seq stop{0, 1};
    rc = (int)cudaMemcpyAsync(&st.seq->stop, &stop.stop, 8, cudaMemcpyHostToDevice, st.s_h);
  }
  if (!rc) rc = (int)cudaStreamSynchronize(st.s_h);
  for (cudaStream_t s : {st.s_base, st.s_idle, st.s_step, st.s_pre})
    if (!rc) rc = (int)cudaStreamSynchronize(s);
  unsigned long long total = 0;
  if (!rc) rc = (int)cudaMemcpy(&total, &st.seq->next, 8, cudaMemcpyDeviceToHost);
  if (!rc) rc = harvest(st, next, cnt, xr, t0, t1, true, total);
  st.err = rc;
}

}  // namespace sq

extern "C" {

// Starts fill-sk16/v2 on `device` over `n` regions (16-byte aligned) under `seed`: BASE kernels of `base_ctas` CTAs
// (0: none) of `warps` warps, `base_budget` units each; epochs of 2^shift units. The IDLE kernels wait for sq_idle(1).
// BASE and IDLE run at stream `priority`, STEP at `step_priority`.
int sq_start(int device, const uint64_t* ptrs, const uint64_t* bytes, int n, const uint8_t* seed, int base_ctas,
             int warps, uint64_t base_budget, int shift, int priority, int step_priority) {
  if (sq::S || n <= 0 || shift < 0 || shift > 20 || (warps != 4 && warps != 8 && warps != 16)) return -1;
  std::vector<tu::Region> regions;
  for (int k = 0; k < n; ++k) regions.push_back(tu::Region{(const uint8_t*)ptrs[k], bytes[k]});
  if (!tu::fill_ok(regions)) return -1;
  auto* st = new sq::State();
  st->device = device;
  st->warps = warps;
  st->base_ctas = base_ctas;
  st->base_budget = base_budget ? base_budget : 4096;
  st->shift = (uint32_t)shift;
  st->plan = tu::fill_regions(regions);
  if (st->plan.first_unit[st->plan.n] == 0) return -1;
  memcpy(st->seed, seed, 32);
  CK(cudaSetDevice(device));
  CK(cudaDeviceGetAttribute(&st->sms, cudaDevAttrMultiProcessorCount, device));
  if (int rc = tu::gate_set(-1, -1, true)) return rc;
  for (cudaStream_t* s : {&st->s_base, &st->s_idle})
    CK(cudaStreamCreateWithPriority(s, cudaStreamNonBlocking, priority));
  CK(cudaStreamCreateWithPriority(&st->s_step, cudaStreamNonBlocking, step_priority));
  CK(cudaStreamCreateWithPriority(&st->s_pre, cudaStreamNonBlocking, step_priority));
  CK(cudaStreamCreateWithFlags(&st->s_h, cudaStreamNonBlocking));
  CK(cudaEventCreate(&st->start));
  CK(cudaEventCreateWithFlags(&st->base_done[0], cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&st->base_done[1], cudaEventDisableTiming));
  CK(cudaEventCreate(&st->idle_done));
  CK(cudaEventCreateWithFlags(&st->step_ev, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&st->step_done, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&st->pre_done, cudaEventDisableTiming));
  CK(cudaMalloc(&st->sg.words, 4ull * fl::SIGN_SLOTS * fl::SIGN_WORDS));
  CK(cudaMalloc(&st->sg.pass, 8 * fl::SIGN_SLOTS));
  CK(cudaMalloc(&st->sg.done, 8 * fl::SIGN_SLOTS));
  CK(cudaMemset(st->sg.pass, 0xFF, 8 * fl::SIGN_SLOTS));
  CK(cudaMemset(st->sg.done, 0, 8 * fl::SIGN_SLOTS));
  for (int k = 0; k < 4; ++k) {
    st->sg.seed[k] = 0;
    for (int b = 0; b < 8; ++b) st->sg.seed[k] = (st->sg.seed[k] << 8) | seed[8 * k + b];
  }
  fl::signs_first<<<1, 32>>>(st->sg);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMalloc(&st->seq, sizeof(fl::Seq)));
  {
    const fl::Seq q0{0, 0, sq::limit(st->shift, 0)};
    CK(cudaMemcpy(st->seq, &q0, sizeof q0, cudaMemcpyHostToDevice));
  }
  CK(cudaMalloc(&st->ring.xr, 4 * sq::EPOCHS));
  CK(cudaMalloc(&st->ring.cnt, 4 * sq::EPOCHS));
  CK(cudaMalloc(&st->ring.t0, 8 * sq::EPOCHS));
  CK(cudaMalloc(&st->ring.t1, 8 * sq::EPOCHS));
  CK(cudaMemset(st->ring.t0, 0xFF, 8 * sq::EPOCHS));
  CK(cudaMemset(st->ring.t1, 0, 8 * sq::EPOCHS));
  CK(cudaMemset(st->ring.xr, 0, 4 * sq::EPOCHS));
  CK(cudaMemset(st->ring.cnt, 0, 4 * sq::EPOCHS));
  st->ring.shift = st->shift;
  st->ring.mask = sq::EPOCHS - 1;
  CK(cudaMalloc(&st->taken, 5 * 8));
  CK(cudaMallocHost(&st->taken_h, 8));
  CK(cudaMalloc(&st->g0, 8));
  CK(cudaEventRecord(st->start, st->s_h));
  fl::stamp<<<1, 1, 0, st->s_h>>>(st->g0);
  CK(cudaMemcpyAsync(&st->g0_h, st->g0, 8, cudaMemcpyDeviceToHost, st->s_h));
  CK(cudaStreamSynchronize(st->s_h));
  st->log.reserve(1 << 20);
  st->run = 1;
  sq::S = st;
  st->th = std::thread(sq::loop, st);
  return 0;
}

// The PREFILL role's reader: 0 the 16-byte loads of every other role, 2–4 the TMA reader with that many stages.
int sq_prefill_reader(int stages) {
  if (!sq::S || (stages != 0 && (stages < 2 || stages > 4))) return -1;
  for (const void* k : {(const void*)fl::fill_seq_bulk_kernel<2>, (const void*)fl::fill_seq_bulk_kernel<3>,
                        (const void*)fl::fill_seq_bulk_kernel<4>, (const void*)fl::fill_seq_bulk_kernel<3, 1>,
                        (const void*)fl::fill_seq_bulk_kernel<3, 2>, (const void*)fl::fill_seq_bulk_kernel<3, 3>})
    if (cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * (int)fl::TB_STAGE))
      return (int)e;
  sq::S->prefill_stages = stages;
  return 0;
}

// STEP's and PREFILL's warps a CTA from now (4, 8 or 16; 0: sq_start's `warps`), and every role's offsets of loads
// in flight a warp (`depth`: 2, or 8 at 4 warps, or 4 or 8 at 8; fill.cu's fill_seq_kernel D). A launch at a pair it
// has no kernel for fails.
int sq_shape(int step_warps, int depth) {
  if (!sq::S || (step_warps != 0 && step_warps != 4 && step_warps != 8 && step_warps != 16) ||
      (depth != 2 && depth != 4 && depth != 8))
    return -1;
  sq::S->step_warps = step_warps;
  sq::S->depth = depth;
  return 0;
}

// A timing probe for every launch from now: fill.cu's NO_HASH (1), NO_MMA (2), both (3) or none (0). Under one, every
// receipt is wrong: for timing only.
int sq_probe(int probe) {
  if (!sq::S || probe < 0 || probe > 3) return -1;
  sq::S->probe = probe;
  return 0;
}

// IDLE kernels on (1) or off (0), `budget` units each (0: the last set, at first 2^22: a kernel returns once a step starts).
void sq_idle(int on, uint64_t budget) {
  if (!sq::S) return;
  if (budget) sq::S->idle_budget = budget;
  sq::S->idle_on = on;
}

namespace sq {

// One kernel of `role` beside the step whose start event is `st.step_ev`, unless the last one (`done`) still runs.
static int beside(State& st, int role, int ctas, cudaStream_t s, cudaEvent_t done, unsigned long long* taken,
                  std::atomic<uint64_t>& launches) {
  if (cudaEventQuery(done) == cudaErrorNotReady) return 0;
  CK(cudaStreamWaitEvent(s, st.step_ev, 0));
  if (int rc = launch(st, role, ctas, taken, st.step_budget, s)) return rc;
  launches += 1;
  return (int)cudaEventRecord(done, s);
}

}  // namespace sq

// A step's start on `stream`: the gate's busy count up (and its prefill count, if the step prefills), and, ordered
// after the counts, a STEP kernel of `ctas` CTAs (if > 0) on the filler's step stream and, for a prefilling step, a
// PREFILL kernel of `prefill_ctas` (if > 0) on its prefill stream. Neither is launched while the last of its role
// still runs: it reads while any such step is on the GPU, so it covers this one, and a launch queued behind it would
// wait, and once the queue is full, block the caller.
int sq_step_begin_phase(int ctas, int prefill, int prefill_ctas, void* stream) {
  sq::State* st = sq::S;
  if (!st) return 0;
  cudaStream_t s = (cudaStream_t)stream;
  fl::gate_step<<<1, 1, 0, s>>>(tu::gate, 1, prefill);
  st->steps += 1;
  if (prefill) st->prefill_steps += 1;
  const bool pre = prefill && prefill_ctas > 0;
  if (ctas <= 0 && !pre) return (int)cudaGetLastError();
  std::lock_guard<std::mutex> g(st->step_mu);
  CK(cudaEventRecord(st->step_ev, s));
  if (ctas > 0)
    if (int rc = sq::beside(*st, fl::STEP, ctas, st->s_step, st->step_done, st->taken + 3, st->step_launches)) return rc;
  if (pre)
    if (int rc = sq::beside(*st, fl::PREFILL, prefill_ctas, st->s_pre, st->pre_done, st->taken + 4, st->prefill_launches))
      return rc;
  return 0;
}

int sq_step_begin(int ctas, void* stream) { return sq_step_begin_phase(ctas, 0, 0, stream); }

int sq_step_end_phase(int prefill, void* stream) {
  if (!sq::S) return 0;
  fl::gate_step<<<1, 1, 0, (cudaStream_t)stream>>>(tu::gate, -1, prefill);
  return (int)cudaGetLastError();
}

int sq_step_end(void* stream) { return sq_step_end_phase(0, stream); }

int sq_stop() {
  sq::State* st = sq::S;
  if (!st) return -1;
  st->run = 0;
  st->th.join();
  int rc = st->err.load();
  tu::gate_set(-1, -1, true);
  for (cudaStream_t s : {st->s_base, st->s_idle, st->s_step, st->s_pre, st->s_h}) cudaStreamDestroy(s);
  for (cudaEvent_t e : {st->start, st->base_done[0], st->base_done[1], st->idle_done, st->step_ev, st->step_done, st->pre_done})
    cudaEventDestroy(e);
  for (void* p : {(void*)st->sg.words, (void*)st->sg.pass, (void*)st->sg.done, (void*)st->seq, (void*)st->ring.xr, (void*)st->ring.cnt, (void*)st->ring.t0, (void*)st->ring.t1,
                  (void*)st->taken, (void*)st->g0})
    cudaFree(p);
  cudaFreeHost(st->taken_h);
  tu::last_log = std::move(st->log);
  delete st;
  sq::S = nullptr;
  return rc;
}

// Bytes and epochs logged, steps begun, STEP kernels launched, prefill steps begun, PREFILL kernels launched, and the
// steps and prefill steps that ended on the GPU with no unit of their role read since they started (fill.cu's Gate).
void sq_progress(uint64_t* out) {
  unsigned int bare[2] = {0, 0};
  if (sq::S && tu::gate) {
    cudaMemcpyAsync(bare, tu::gate->bare, sizeof bare, cudaMemcpyDeviceToHost, sq::S->s_h);
    cudaStreamSynchronize(sq::S->s_h);
  }
  out[6] = bare[0];
  out[7] = bare[1];
  out[0] = sq::S ? sq::S->units_done.load() * fl::UNIT_BYTES : 0;
  out[1] = sq::S ? sq::S->epochs_done.load() : 0;
  out[2] = sq::S ? sq::S->steps.load() : 0;
  out[3] = sq::S ? sq::S->step_launches.load() : 0;
  out[4] = sq::S ? sq::S->prefill_steps.load() : 0;
  out[5] = sq::S ? sq::S->prefill_launches.load() : 0;
}

// An event recorded now on `stream`, on the start event's clock (ms), or -1.
double sq_now_ms(void* stream) {
  sq::State* st = sq::S;
  if (!st) return -1;
  cudaEvent_t e;
  if (cudaEventCreate(&e) != cudaSuccess) return -1;
  float ms = -1;
  if (cudaEventRecord(e, (cudaStream_t)stream) == cudaSuccess && cudaEventSynchronize(e) == cudaSuccess)
    cudaEventElapsedTime(&ms, st->start, e);
  cudaEventDestroy(e);
  return ms;
}

// The running v2 filler's log, as tu_log reads a stopped run's (the stopped one's is tu_log's).
int sq_log(uint64_t first, double* out, int max) {
  sq::State* st = sq::S;
  std::unique_lock<std::mutex> g;
  if (st) g = std::unique_lock<std::mutex>(st->m);
  const std::vector<tu::Rec>& log = st ? st->log : tu::last_log;
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

}  // extern "C"
