// The receipted filler (filler.cu's base-u1 kernel, FT = 4 m16n8k32 s8 tiles a warp with duo-φ receipts at L = 32) as a
// library for serve.py: a native pump thread (no GIL, beside an engine that runs Python) keeps `depth` chunks of the
// filler queued on a stream, each `grid` blocks of `fw` filler warps running `iters` rounds over operands salted by the
// chunk's seed (seed0 + chunk index), so a device-wide synchronize in the engine waits for at most `depth` chunks.
// Each chunk's per-warp clock64 spans land in mapped host memory, so the pump adds up the filler's MMAs and its blocks'
// SM-cycles without a copy on any stream the engine uses.
//
// Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -o libifill.so serve.cu
#define FILLER_LIB
#include "filler.cu"

#include <atomic>
#include <thread>

namespace {
const KFn FILL = fk<O_NONE, O_NONE, 0, 1, 1>;
struct Slot {
  uint32_t* out = nullptr;
  long long* fclk = nullptr;  // mapped host memory, its device alias in fclk_d
  long long* fclk_d = nullptr;
  uint32_t* smid = nullptr;
  uint32_t* smid_d = nullptr;
  cudaEvent_t ev{};
  long long k = -1;
};
std::vector<Slot> slots;
long long* pinfo = nullptr;
int* gdone = nullptr;
int g_grid = 0, g_fw = 0, g_iters = 0, g_depth = 0, g_smem = 0, g_rc = 0;
uint32_t g_seed0 = 0;
long long g_last = -1;
std::thread pump;
std::atomic<bool> on{false};
// chunks done, filler warp-MMAs done, the blocks' summed SM-cycle spans, blocks done, the most blocks one SM ran in a chunk
std::atomic<long long> st_chunks{0}, st_mmas{0}, st_span{0}, st_blocks{0}, st_stack{0};

void harvest(const Slot& s) {
  long long span = 0;
  std::vector<int> per(NSMID, 0);
  int stack = 0;
  for (int b = 0; b < g_grid; ++b) {
    long long lo = 0, hi = 0;
    for (int w = 0; w < g_fw; ++w) {
      const long long t0 = s.fclk[2 * (b * MAXW + w)], t1 = s.fclk[2 * (b * MAXW + w) + 1];
      lo = w == 0 ? t0 : std::min(lo, t0);
      hi = w == 0 ? t1 : std::max(hi, t1);
    }
    span += hi - lo;
    if (s.smid[b] < (uint32_t)NSMID) stack = std::max(stack, ++per[s.smid[b]]);
  }
  st_span += span;
  st_blocks += g_grid;
  st_mmas += (long long)g_grid * g_fw * g_iters * FT;
  st_chunks += 1;
  if (stack > st_stack.load()) st_stack = stack;
}
}  // namespace

// Buffers for `depth` chunks of `grid` blocks, the receipt keys from kseed. Returns 0 or a cudaError_t.
extern "C" int ifill_init(uint32_t kseed, int grid, int depth) {
  if (!slots.empty() || grid < 1 || grid > 1024 || depth < 1) return -1;
  int e;
  set_keys(kseed);
  if ((e = cudaMalloc(&pinfo, (size_t)grid * MAXW * 4 * sizeof(long long)))) return e;
  if ((e = cudaMalloc(&gdone, 2 * NSMID * sizeof(int)))) return e;
  if ((e = cudaMemset(gdone, 0, 2 * NSMID * sizeof(int)))) return e;
  slots.resize(depth);
  for (Slot& s : slots) {
    if ((e = cudaMalloc(&s.out, (size_t)grid * MAXW * 32 * OUTW * sizeof(uint32_t)))) return e;
    if ((e = cudaHostAlloc(&s.fclk, (size_t)grid * MAXW * 2 * sizeof(long long), cudaHostAllocMapped))) return e;
    if ((e = cudaHostGetDevicePointer(&s.fclk_d, s.fclk, 0))) return e;
    if ((e = cudaHostAlloc(&s.smid, (size_t)grid * sizeof(uint32_t), cudaHostAllocMapped))) return e;
    if ((e = cudaHostGetDevicePointer(&s.smid_d, s.smid, 0))) return e;
    if ((e = cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming | cudaEventBlockingSync))) return e;
  }
  return cudaDeviceSynchronize();
}

// A non-blocking stream at `prio` (cudaDeviceGetStreamPriorityRange's numbers: 0 the lowest, negative higher).
extern "C" int ifill_stream(int prio, void** s) {
  cudaStream_t st;
  const int e = cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, prio);
  *s = st;
  return e;
}

// The pump on `stream`: grid blocks of fw filler warps a chunk, iters rounds each (a multiple of 1), smem bytes of dynamic
// shared memory a block (0: none, so a block takes only registers and warp slots from the SM).
extern "C" int ifill_start(int grid, int fw, int iters, int smem, void* stream, uint32_t seed0) {
  if (slots.empty() || on.load() || grid < 1 || fw < 1 || fw > MAXW || iters < 1) return -1;
  if (smem > 0 && cudaFuncSetAttribute(FILL, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return -2;
  g_grid = grid, g_fw = fw, g_iters = iters, g_smem = smem, g_seed0 = seed0, g_depth = (int)slots.size(), g_rc = 0;
  g_last = -1;
  st_chunks = st_mmas = st_span = st_blocks = st_stack = 0;
  cudaStream_t s = (cudaStream_t)stream;
  on = true;
  pump = std::thread([s]() {
    long long k = 0;
    for (; on.load(); ++k) {
      Slot& sl = slots[k % g_depth];
      if (sl.k >= 0) {
        cudaEventSynchronize(sl.ev);
        harvest(sl);
        sl.k = -1;
      }
      FILL<<<g_grid, 32 * g_fw, g_smem, s>>>(sl.out, sl.fclk_d, pinfo, sl.smid_d, gdone, g_iters, g_fw, 0,
                                              g_seed0 + (uint32_t)k);
      if ((g_rc = (int)cudaGetLastError())) break;
      cudaEventRecord(sl.ev, s);
      sl.k = k;
      g_last = k;
    }
    for (long long j = k - g_depth; j < k; ++j) {
      if (j < 0) continue;
      Slot& sl = slots[j % g_depth];
      if (sl.k == j) {
        cudaEventSynchronize(sl.ev);
        harvest(sl);
        sl.k = -1;
      }
    }
  });
  return 0;
}

// o: chunks, warp-MMAs, summed block spans (SM-cycles), blocks, most blocks on one SM in a chunk (all done so far).
extern "C" void ifill_stats(long long* o) {
  o[0] = st_chunks.load();
  o[1] = st_mmas.load();
  o[2] = st_span.load();
  o[3] = st_blocks.load();
  o[4] = st_stack.load();
}

// Stops launching, waits for the chunks in flight. Returns the chunks done; *rc the first launch error.
extern "C" long long ifill_stop(int* rc) {
  on = false;
  if (pump.joinable()) pump.join();
  if (rc) *rc = g_rc;
  return st_chunks.load();
}

// After ifill_stop: the last chunk's first n filler threads' OUTW words (ftid order), its seed and iters.
extern "C" int ifill_last(uint32_t* host, int n, uint32_t* seed, int* iters) {
  if (on.load() || g_last < 0 || n > g_grid * g_fw * 32) return -1;
  const Slot& sl = slots[g_last % g_depth];
  *seed = g_seed0 + (uint32_t)g_last;
  *iters = g_iters;
  return (int)cudaMemcpy(host, sl.out, (size_t)n * OUTW * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

// One block's dynamic shared memory that keeps one filler block an SM (the device's per-block maximum, less the static
// word), for spatial arms that should not stack blocks.
extern "C" int ifill_smem_one() {
  int dev = 0, per_smp = 0, reserved = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&per_smp, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  return std::min(per_smp - reserved, optin) - 256;
}
