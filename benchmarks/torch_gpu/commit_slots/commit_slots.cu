// The serve-time commit call's kernels (commit_slots.cuh), launched on the caller's stream; built by slots_gpu.device_lib
// with nvcc. Every launch function returns 0, or the CUDA error of its launch.
#include <cuda_runtime.h>

#include "commit_slots.cuh"

using namespace cs;

namespace {

constexpr int THREADS = 256;
constexpr int TSMEM = 256 * TSTRIDE;

__global__ void __launch_bounds__(THREADS) k_rows(Ctx c, Step st, Grp g) {
  extern __shared__ uint4 smem[];
  const uint4* Tg = reinterpret_cast<const uint4*>(c.T);
  for (int i = threadIdx.x; i < 256 * 16; i += blockDim.x) smem[(i >> 4) * (TSTRIDE / 16) + (i & 15)] = Tg[i];
  __syncthreads();
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)g.nrows * st.S) return;
  row_thread(c, st, g, (u32)(t / st.S), (u32)(t % st.S), reinterpret_cast<const uint8_t*>(smem), TSTRIDE);
}

__global__ void __launch_bounds__(THREADS) k_units(Ctx c, Step st, Grp g) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)g.nunits * st.S) return;
  unit_thread(c, st, g, (u32)(t / st.S), (u32)(t % st.S));
}

__global__ void __launch_bounds__(THREADS) k_fold(Ctx c, Step st, Grp g) {
  const u64 s = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (s < st.S) fold_thread(c, st, g, (u32)s);
}

__global__ void __launch_bounds__(THREADS) k_finish(Ctx c, Step st) {
  const u64 s = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (s < st.S) finish_thread(c, st, (u32)s);
}

__global__ void __launch_bounds__(THREADS) k_step_pads(Step st) {
  const u64 i = (u64)st.S + (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < st.W) step_pad_thread(st, (u32)i);
}

__global__ void __launch_bounds__(THREADS) k_level(Step st, const u64* in, u64* out, int l) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (st.W >> (l + 1))) level_thread(st, in, out, l, (u32)i);
}

unsigned blocks(u64 n) { return (unsigned)((n + THREADS - 1) / THREADS); }

int launched() { return (int)cudaGetLastError(); }

}  // namespace

extern "C" {

void cs_midstate(const uint8_t* block, uint64_t* out) { midstate(block, out); }

int cs_tail(const uint64_t* did, uint64_t n, int depth, uint32_t* lv, uint64_t* nodes) { return tail(did, n, depth, lv, nodes); }

int cs_rows(void* stream, const Ctx* c, const Step* st, const Grp* g) {
  static bool ready = false;
  if (!ready) {
    const cudaError_t e = cudaFuncSetAttribute(k_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, TSMEM);
    if (e != cudaSuccess) return (int)e;
    ready = true;
  }
  const u64 n = (u64)g->nrows * st->S;
  if (n) k_rows<<<blocks(n), THREADS, TSMEM, (cudaStream_t)stream>>>(*c, *st, *g);
  return launched();
}

int cs_units(void* stream, const Ctx* c, const Step* st, const Grp* g) {
  const u64 n = (u64)g->nunits * st->S;
  if (n) k_units<<<blocks(n), THREADS, 0, (cudaStream_t)stream>>>(*c, *st, *g);
  return launched();
}

int cs_fold(void* stream, const Ctx* c, const Step* st, const Grp* g) {
  k_fold<<<blocks(st->S), THREADS, 0, (cudaStream_t)stream>>>(*c, *st, *g);
  return launched();
}

int cs_finish(void* stream, const Ctx* c, const Step* st) {
  cudaStream_t s = (cudaStream_t)stream;
  k_finish<<<blocks(st->S), THREADS, 0, s>>>(*c, *st);
  if (st->W > st->S) k_step_pads<<<blocks(st->W - st->S), THREADS, 0, s>>>(*st);
  u64 *in = st->lv_a, *out = st->lv_b;
  for (u32 l = 0; l < st->depth; ++l) {
    k_level<<<blocks(st->W >> (l + 1)), THREADS, 0, s>>>(*st, in, out, (int)l);
    u64* t = in; in = out; out = t;
  }
  const cudaError_t e = cudaMemcpyAsync(st->root, in, 64, cudaMemcpyDeviceToDevice, s);
  return e != cudaSuccess ? (int)e : launched();
}

}
