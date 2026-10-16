// COMMIT-GPU's committer (benchmarks/commit_gpu/commit_gpu.cuh) for wgemm.cu's pipe mode, in a translation unit of its own
// (its __constant__ C and its names stay apart from the GEMM's): a call's activation operand X (m x k int8) as decision
// 46's tile leaves (64 rows x 32 values) and its checkpointed receipts as 2 KiB rows (one window of one tile a row), the salt
// phase and the data phase each one CUDA graph, the two trees its parallel branches (each tree's narrow top levels are
// launch-latency bound, so they overlap). Both trees run under one params file's constants: both rows are 16,384 bits, so
// the work is the same as under each tree's own domain.
//
// By default on the committer's small path (CTAs of 128 threads with no shared memory, which fit beside other CTAs on an
// SM; the X tree in domain slot 0, the receipts' in slot 1), its kernels preferring a 100% shared-memory carveout as the
// GEMM does, so an SM needn't drain to switch between them. env: CG_SMALL (1; 0: the default path), CG_CARVE (100; -1:
// the driver's).
#define COMMIT_GPU_DOMS 2
#include "../commit_gpu/commit_gpu.cuh"

#include <functional>

namespace {
struct CgPipe {
  Committer x, r;
  cudaGraphExec_t salts, data;
};
// one graph, fx on st and fr on a branch forked from it and joined back
cudaGraphExec_t capture(cudaStream_t st, cudaStream_t st2, const std::function<void()>& fx, const std::function<void()>& fr) {
  cudaGraph_t g;
  cudaGraphExec_t e;
  cudaEvent_t fork, join;
  CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  CK(cudaEventRecord(fork, st));
  CK(cudaStreamWaitEvent(st2, fork));
  fx();
  fr();
  CK(cudaEventRecord(join, st2));
  CK(cudaStreamWaitEvent(st, join));
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&e, g, 0));
  return e;
}
int env_int(const char* n, int d) { return getenv(n) ? atoi(getenv(n)) : d; }
}  // namespace

bool cgp_small() { return env_int("CG_SMALL", 1) != 0; }
void* cgp_new(const char* params, const uint8_t* X, uint64_t m, uint64_t k, const uint8_t* R, uint64_t rows) {
  const Params p = read_params(params);
  const bool small = cgp_small();
  const int carve = env_int("CG_CARVE", 100);
  if (small && carve >= 0) {
    const void* f[] = {(const void*)salt_small<true>, (const void*)leaf_small<RowTile, 0>, (const void*)leaf_small<RowBytes, 0>,
                       (const void*)level_kernel, (const void*)top_small};
    for (const void* g : f) CK(cudaFuncSetAttribute(g, cudaFuncAttributePreferredSharedMemoryCarveout, carve));
  }
  auto* c = new CgPipe{};
  c->x.init(p, tile_leaves(m, k, 64), 2048, false, false, 0, 0, small);
  c->r.init(p, rows, 2048, false, false, 0, small ? 1 : 0, small);
  cudaStream_t st, st2;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking));
  c->salts = capture(st, st2, [&] { c->x.salts<true>(nullptr, st); }, [&] { c->r.salts<true>(nullptr, st2); });
  c->data = capture(
      st, st2, [&] { c->x.tiles(X, m, k, 64, false, st); c->x.nodes(st); },
      [&] { c->r.leaves<RowBytes>(RowBytes{R, 2048, 2048}, st2); c->r.nodes(st2); });
  CK(cudaGraphLaunch(c->salts, st));
  CK(cudaGraphLaunch(c->data, st));
  CK(cudaStreamSynchronize(st));
  CK(cudaGetLastError());
  CK(cudaStreamDestroy(st));
  CK(cudaStreamDestroy(st2));
  return c;
}
void cgp_salts(void* h, cudaStream_t st) { CK(cudaGraphLaunch(static_cast<CgPipe*>(h)->salts, st)); }
void cgp_data(void* h, cudaStream_t st) { CK(cudaGraphLaunch(static_cast<CgPipe*>(h)->data, st)); }
