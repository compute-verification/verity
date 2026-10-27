// Does a wide decode kept apart from the GEMM overlap the weights' read? sqfloor.cu's 1,536-bit squaring (form 2, one
// level of Karatsuba, 252 registers a thread, so two 128-thread blocks fill an SM's register file) beside a streaming
// read of HBM, on two streams. ra-codec-theory's 1,536-bit point prices its decode at this squaring's standalone rate
// (0.93 passes of the read) and charges only the excess over the read as step time, which needs the two at once.
//
// The squaring work is fixed at RATIO of the read's time at the squaring's full-occupancy rate, then placed SQ_BPSM
// blocks a SM (so one block a SM squares twice as long a chain); the read is RD_BPSM 256-thread blocks a SM.
//
//   sqoverlap gate K N                          N words, K squarings, in sqfloor's gate format (sqfloor_gate.py)
//   sqoverlap run SQ_BPSM RD_BPSM RATIO REPS    one JSON line: each alone, and both on two streams in each order
#define SQFLOOR_NO_MAIN
#include "sqfloor.cu"

namespace {

constexpr size_t BUF_BYTES = 4ull << 30;
constexpr int PASSES = 8;

__global__ void __launch_bounds__(256) stream_read(const uint4 *__restrict__ buf, size_t nvec, int passes,
                                                   uint32_t *out) {
  uint32_t acc = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int p = 0; p < passes; ++p) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < nvec; i += 4 * stride) {
      const uint4 a = __ldcs(buf + i), b = __ldcs(buf + i + stride), c = __ldcs(buf + i + 2 * stride),
                  d = __ldcs(buf + i + 3 * stride);
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < nvec; i += stride) {
      const uint4 a = __ldcs(buf + i);
      acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
  }
  if (acc == 0x9e3779b9u) out[blockIdx.x] = acc;
}

struct Both {
  float total, sq_end, rd_end;
};

float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  if (!strcmp(argv[1], "gate") && argc == 4) {
    const int K = atoi(argv[2]), n = atoi(argv[3]);
    std::vector<uint32_t> h0, h;
    fill(h0, n);
    uint32_t *d;
    CK(cudaMalloc(&d, h0.size() * 4));
    CK(cudaMemcpy(d, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice));
    sq::chain<2><<<(n + 127) / 128, 128>>>(d, n, K);
    CK(cudaDeviceSynchronize());
    h.resize(h0.size());
    CK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
    for (int w = 0; w < n; ++w) {
      printf("2 %d %d %u ", K, SQ_BITS, sq::CF);
      print_canon(h0, n, w);
      printf(" ");
      print_canon(h, n, w);
      printf("\n");
    }
    return 0;
  }
  if (strcmp(argv[1], "run") || argc != 6) return 2;
  const int sq_bpsm_req = atoi(argv[2]), rd_bpsm = atoi(argv[3]), reps = atoi(argv[5]);
  const double ratio = atof(argv[4]);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  const int sms = prop.multiProcessorCount;
  int sq_max = 0, rd_max = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sq_max, sq::chain<2>, 128, 0));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&rd_max, stream_read, 256, 0));
  cudaFuncAttributes sq_attr, rd_attr;
  CK(cudaFuncGetAttributes(&sq_attr, sq::chain<2>));
  CK(cudaFuncGetAttributes(&rd_attr, stream_read));
  const int sq_bpsm = std::min(sq_bpsm_req, sq_max);

  uint4 *buf;
  uint32_t *sink;
  CK(cudaMalloc(&buf, BUF_BYTES));
  CK(cudaMemset(buf, 0x5a, BUF_BYTES));
  CK(cudaMalloc(&sink, 1 << 20));
  const size_t nvec = BUF_BYTES / sizeof(uint4);
  const int rd_grid = sms * rd_bpsm;

  const int n_full = sms * sq_max * 128, n = sms * sq_bpsm * 128;
  std::vector<uint32_t> h;
  fill(h, n_full);
  uint32_t *xs;
  CK(cudaMalloc(&xs, h.size() * 4));
  CK(cudaMemcpy(xs, h.data(), h.size() * 4, cudaMemcpyHostToDevice));

  cudaStream_t s_sq, s_rd;
  CK(cudaStreamCreateWithFlags(&s_sq, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s_rd, cudaStreamNonBlocking));
  cudaEvent_t t0, t_sq, t_rd;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t_sq));
  CK(cudaEventCreate(&t_rd));

  auto alone_rd = [&]() {
    CK(cudaEventRecord(t0, s_rd));
    stream_read<<<rd_grid, 256, 0, s_rd>>>(buf, nvec, PASSES, sink);
    CK(cudaEventRecord(t_rd, s_rd));
    CK(cudaEventSynchronize(t_rd));
    float ms;
    CK(cudaEventElapsedTime(&ms, t0, t_rd));
    return ms;
  };
  auto alone_sq = [&](int words, int K) {
    CK(cudaEventRecord(t0, s_sq));
    sq::chain<2><<<words / 128, 128, 0, s_sq>>>(xs, words, K);
    CK(cudaEventRecord(t_sq, s_sq));
    CK(cudaEventSynchronize(t_sq));
    float ms;
    CK(cudaEventElapsedTime(&ms, t0, t_sq));
    return ms;
  };
  auto both = [&](int K, bool sq_first) {
    CK(cudaEventRecord(t0, s_sq));
    CK(cudaStreamWaitEvent(s_rd, t0, 0));
    if (sq_first) {
      sq::chain<2><<<n / 128, 128, 0, s_sq>>>(xs, n, K);
      stream_read<<<rd_grid, 256, 0, s_rd>>>(buf, nvec, PASSES, sink);
    } else {
      stream_read<<<rd_grid, 256, 0, s_rd>>>(buf, nvec, PASSES, sink);
      sq::chain<2><<<n / 128, 128, 0, s_sq>>>(xs, n, K);
    }
    CK(cudaEventRecord(t_sq, s_sq));
    CK(cudaEventRecord(t_rd, s_rd));
    CK(cudaEventSynchronize(t_sq));
    CK(cudaEventSynchronize(t_rd));
    float a, b;
    CK(cudaEventElapsedTime(&a, t0, t_sq));
    CK(cudaEventElapsedTime(&b, t0, t_rd));
    return Both{std::max(a, b), a, b};
  };

  alone_rd();
  alone_sq(n_full, 200);
  std::vector<float> rd_ms, sq_full_ms;
  for (int r = 0; r < reps; ++r) rd_ms.push_back(alone_rd());
  const float t_read = median(rd_ms);
  const int k_cal = 1000;
  for (int r = 0; r < reps; ++r) sq_full_ms.push_back(alone_sq(n_full, k_cal));
  const double rate_full = (double)n_full * k_cal / (median(sq_full_ms) * 1e-3);
  const double squarings = ratio * t_read * 1e-3 * rate_full;
  const int K = std::max(1, (int)(squarings / n + 0.5));

  std::vector<float> sq_ms, both_sq_total, both_sq_sq, both_sq_rd, both_rd_total, both_rd_sq, both_rd_rd;
  for (int r = 0; r < reps; ++r) {
    sq_ms.push_back(alone_sq(n, K));
    Both x = both(K, true);
    both_sq_total.push_back(x.total), both_sq_sq.push_back(x.sq_end), both_sq_rd.push_back(x.rd_end);
    Both y = both(K, false);
    both_rd_total.push_back(y.total), both_rd_sq.push_back(y.sq_end), both_rd_rd.push_back(y.rd_end);
  }
  const float t_sq_alone = median(sq_ms);
  const double gbps = (double)BUF_BYTES * PASSES / (t_read * 1e-3) / 1e9;
  auto overlap = [&](float t) { return (t_read + t_sq_alone - t) / std::min(t_read, t_sq_alone); };
  const float ts = median(both_sq_total), tr = median(both_rd_total);
  printf("{\"bits\": %d, \"sms\": %d, \"sq_regs\": %d, \"sq_blocks_per_sm_max\": %d, \"sq_blocks_per_sm\": %d, "
         "\"rd_regs\": %d, \"rd_blocks_per_sm_max\": %d, \"rd_blocks_per_sm\": %d, \"ratio\": %.3f, \"words\": %d, "
         "\"K\": %d, \"sq_rate_full_occupancy\": %.4e, \"sq_rate_alone\": %.4e, \"read_ms\": %.3f, \"read_gbps\": %.1f, "
         "\"sq_alone_ms\": %.3f, \"both_sq_first_ms\": %.3f, \"both_sq_first_sq_end_ms\": %.3f, "
         "\"both_sq_first_rd_end_ms\": %.3f, \"both_rd_first_ms\": %.3f, \"both_rd_first_sq_end_ms\": %.3f, "
         "\"both_rd_first_rd_end_ms\": %.3f, \"overlap_sq_first\": %.3f, \"overlap_rd_first\": %.3f, "
         "\"step_over_read_sq_first\": %.3f, \"step_over_read_rd_first\": %.3f}\n",
         SQ_BITS, sms, sq_attr.numRegs, sq_max, sq_bpsm, rd_attr.numRegs, rd_max, rd_bpsm, ratio, n, K, rate_full,
         (double)n * K / (t_sq_alone * 1e-3), t_read, gbps, t_sq_alone, ts, median(both_sq_sq), median(both_sq_rd), tr,
         median(both_rd_sq), median(both_rd_rd), overlap(ts), overlap(tr), ts / t_read, tr / t_read);
  return 0;
}
