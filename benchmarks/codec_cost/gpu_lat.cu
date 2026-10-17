// One word's dependent squaring chain on one GPU thread: sq.cu's `chain` kernel (benchmarks/rate_squaring) launched
// as one block of one thread, so the time a squaring is its latency, not a share of the GPU's throughput. Build at the
// widths sq.cu takes (-DSQ_BITS=768 -DSQ_C=825, ...).
//
//   gpu_lat FOLD K REPS   JSON: ns a squaring (min and median of REPS launches of K squarings)
#define SQ_NO_MAIN
#include "../rate_squaring/sq.cu"

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  const int fold = atoi(argv[1]), K = atoi(argv[2]), reps = atoi(argv[3]);
  std::vector<uint32_t> h;
  fill(h, 1);
  uint32_t *d;
  CK(cudaMalloc(&d, h.size() * 4));
  CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  Kernel k = kernel(fold, 1);
  k<<<1, 1>>>(d, 1, K / 10 + 1);
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  std::vector<double> ns;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(a));
    k<<<1, 1>>>(d, 1, K);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    ns.push_back(ms * 1e6 / K);
  }
  std::sort(ns.begin(), ns.end());
  cudaFuncAttributes attr;
  CK(cudaFuncGetAttributes(&attr, k));
  printf("{\"bits\": %d, \"c\": %u, \"fold\": %d, \"K\": %d, \"reps\": %d, \"regs\": %d, \"local_bytes\": %zu, "
         "\"sqr_min_ns\": %.2f, \"sqr_med_ns\": %.2f}\n",
         SQ_BITS, sq::CF, fold, K, reps, attr.numRegs, attr.localSizeBytes, ns.front(), ns[ns.size() / 2]);
  return 0;
}
