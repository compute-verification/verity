// COMMIT-GPU's committer (benchmarks/commit_gpu/commit_gpu.cuh) for wgemm.cu's pipe mode, in a translation unit of its own
// (its __constant__ C and its names stay apart from the GEMM's): a call's activation operand X (m x k int8) as decision
// 46's tile leaves (64 rows x 32 values) and its checkpointed receipts as 2 KiB rows (one window of one tile a row), the salt
// phase and the data phase each one CUDA graph. Both trees run under one params file's constants: both rows are 16,384
// bits, so the work is the same as under each tree's own domain.
#include "../commit_gpu/commit_gpu.cuh"

#include <functional>

namespace {
struct CgPipe {
  Committer x, r;
  cudaGraphExec_t salts, data;
};
cudaGraphExec_t capture(cudaStream_t st, const std::function<void()>& f) {
  cudaGraph_t g;
  cudaGraphExec_t e;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  f();
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&e, g, 0));
  return e;
}
}  // namespace

void* cgp_new(const char* params, const uint8_t* X, uint64_t m, uint64_t k, const uint8_t* R, uint64_t rows) {
  const Params p = read_params(params);
  auto* c = new CgPipe{};
  c->x.init(p, tile_leaves(m, k, 64), 2048, false);
  c->r.init(p, rows, 2048, false);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  c->salts = capture(st, [&] { c->x.salts<true>(nullptr, st); c->r.salts<true>(nullptr, st); });
  c->data = capture(st, [&] {
    c->x.tiles(X, m, k, 64, false, st);
    c->x.nodes(st);
    c->r.leaves<RowBytes>(RowBytes{R, 2048, 2048}, st);
    c->r.nodes(st);
  });
  CK(cudaGraphLaunch(c->salts, st));
  CK(cudaGraphLaunch(c->data, st));
  CK(cudaStreamSynchronize(st));
  CK(cudaGetLastError());
  CK(cudaStreamDestroy(st));
  return c;
}
void cgp_salts(void* h, cudaStream_t st) { CK(cudaGraphLaunch(static_cast<CgPipe*>(h)->salts, st)); }
void cgp_data(void* h, cudaStream_t st) { CK(cudaGraphLaunch(static_cast<CgPipe*>(h)->data, st)); }
