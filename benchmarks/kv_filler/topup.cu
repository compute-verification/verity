// The filler top-up: a host thread of its own reads registered regions of HBM on a low-priority stream, chunk after
// chunk, each chunk with its receipts, so that a window's reads approach R·T while a serving process runs beside it and
// its Python never waits on the top-up.
//
//   mode -1  read only: every 16 bytes loaded once and XOR-folded (the floor of a receipt's cost)
//   mode  0  gf31-poly/v1 over 2,056-byte blocks, the bandwidth audit's uncertified fast check
//   mode  2  ro-blake3/v1 over 2,056-byte blocks, the bandwidth audit's certified receipt, with its leaves and tree
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

constexpr int MODE_READ = -1;

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
    bw::Pass pass = bw::make_pass(st.mode, blocks, st.ws[i]);
    uint8_t seed[32];
    memcpy(seed, st.seed, 32);
    for (int k = 0; k < 8; ++k) seed[k] ^= (uint8_t)(st.chunk >> (8 * k));
    bw::launch_tags(pass, p, seed, blocks, st.s);
    bw::launch_tree(pass, st.s);
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
  if (tu::S || n <= 0 || depth <= 0 || (mode != tu::MODE_READ && mode != bw::TAG_GF31 && mode != bw::TAG_B3)) return -1;
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
    if (mode >= 0) CK(cudaMalloc(&st->ws[i], bw::workspace_bytes(mode, chunk_bytes / bw::BLOCK_BYTES)));
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

// A plain read of `bytes` at `p` by `grid` CTAs of `tile_bytes` tiles (grid 0: one CTA a tile), for R.
int tu_read(const void* p, uint64_t bytes, uint64_t tile_bytes, void* stream, uint32_t* xr) {
  const uint64_t n16 = bytes / 16, tile16 = tile_bytes / 16;
  tu::tile_read<<<(unsigned)((n16 + tile16 - 1) / tile16), 256, 0, (cudaStream_t)stream>>>((const uint4*)p, n16,
                                                                                         (uint32_t)tile16, xr);
  return (int)cudaGetLastError();
}

}  // extern "C"
