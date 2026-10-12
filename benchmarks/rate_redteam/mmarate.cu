// What can the tensor cores add to an adversary's squaring? The u8 mma.m16n8k32's rate on sm_120, alone and beside
// sqfloor.cu's 1,536-bit squaring (form 2, one level of Karatsuba) on another stream. The operands stay in registers,
// so the rate is a ceiling on the MMA a tensor-core squaring could issue beside a CUDA-core one; a real one also loads
// its Toeplitz fragments and carries its coefficients.
//
//   mmarate alone WARPS BPSM CH REPS            one JSON line: MMAs a second, WARPS-warp blocks, BPSM a SM, CH chains
//   mmarate both SQ_BPSM WARPS BPSM CH REPS     one JSON line: each alone, then both on two streams in each order, the
//                                               squaring's work and the MMAs' each sized to about 25 ms alone
#define SQFLOOR_NO_MAIN
#include "sqfloor.cu"
    10|
namespace {

template <int CH>
__global__ void mma_loop(uint64_t iters, uint32_t seed, uint32_t *out) {
  uint32_t a[4], b[2], c[CH][4];
  const uint32_t t = threadIdx.x * 0x9E3779B9u ^ blockIdx.x * 0x85EBCA6Bu ^ seed;
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = t * (2 * i + 1) ^ 0x5bd1e995u;
  b[0] = t ^ 0xc2b2ae35u;
  b[1] = t * 0x27d4eb2fu;
#pragma unroll
  for (int k = 0; k < CH; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[k][i] = k + i;
  for (uint64_t it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CH; ++k)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                   "{%0,%1,%2,%3};"
                   : "+r"(c[k][0]), "+r"(c[k][1]), "+r"(c[k][2]), "+r"(c[k][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  uint32_t acc = 0;
#pragma unroll
  for (int k = 0; k < CH; ++k) acc ^= c[k][0] ^ c[k][1] ^ c[k][2] ^ c[k][3];
  if (acc == 0x9e3779b9u) out[blockIdx.x] = acc;
}

template <int CH>
void launch(int grid, int threads, uint64_t iters, uint32_t *sink, cudaStream_t s) {
  mma_loop<CH><<<grid, threads, 0, s>>>(iters, 1u, sink);
}

void launch_ch(int ch, int grid, int threads, uint64_t iters, uint32_t *sink, cudaStream_t s) {
  switch (ch) {
    case 1: launch<1>(grid, threads, iters, sink, s); break;
    case 2: launch<2>(grid, threads, iters, sink, s); break;
    case 4: launch<4>(grid, threads, iters, sink, s); break;
    case 8: launch<8>(grid, threads, iters, sink, s); break;
    default: fprintf(stderr, "CH is 1, 2, 4 or 8\n"); exit(2);
  }
}

int regs_ch(int ch) {
  cudaFuncAttributes a;
  switch (ch) {
    case 1: CK(cudaFuncGetAttributes(&a, mma_loop<1>)); break;
    case 2: CK(cudaFuncGetAttributes(&a, mma_loop<2>)); break;
    case 4: CK(cudaFuncGetAttributes(&a, mma_loop<4>)); break;
    default: CK(cudaFuncGetAttributes(&a, mma_loop<8>)); break;
  }
  return a.numRegs;
}

float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Timer {
  cudaEvent_t a, b;
  Timer() {
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
  }
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  const int sms = prop.multiProcessorCount;
  uint32_t *sink;
  CK(cudaMalloc(&sink, 1 << 20));
  cudaStream_t s_sq, s_mm;
  CK(cudaStreamCreateWithFlags(&s_sq, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s_mm, cudaStreamNonBlocking));
  Timer tm, ts;
  cudaEvent_t t0;
  CK(cudaEventCreate(&t0));

  auto time_mma = [&](int ch, int grid, int threads, uint64_t iters) {
    CK(cudaEventRecord(tm.a, s_mm));
    launch_ch(ch, grid, threads, iters, sink, s_mm);
    CK(cudaEventRecord(tm.b, s_mm));
    CK(cudaEventSynchronize(tm.b));
    float ms;
    CK(cudaEventElapsedTime(&ms, tm.a, tm.b));
    return ms;
  };

  if (!strcmp(argv[1], "alone") && argc == 6) {
    const int warps = atoi(argv[2]), bpsm = atoi(argv[3]), ch = atoi(argv[4]), reps = atoi(argv[5]);
    const int grid = sms * bpsm, threads = 32 * warps;
    time_mma(ch, grid, threads, 100);
    const float cal = time_mma(ch, grid, threads, 2000);
    const uint64_t iters = std::max<uint64_t>(100, (uint64_t)(2000.0 * 25.0 / cal));
    std::vector<float> v;
    for (int r = 0; r < reps; ++r) v.push_back(time_mma(ch, grid, threads, iters));
    const double mmas = (double)grid * warps * ch * iters, ms = median(v);
    printf("{\"mode\": \"alone\", \"sms\": %d, \"warps\": %d, \"blocks_per_sm\": %d, \"chains\": %d, \"regs\": %d, "
           "\"iters\": %llu, \"ms\": %.3f, \"mma_per_s\": %.4e, \"mac_per_s\": %.4e}\n",
           sms, warps, bpsm, ch, regs_ch(ch), (unsigned long long)iters, ms, mmas / (ms * 1e-3),
           mmas * 4096 / (ms * 1e-3));
    return 0;
  }
  if (strcmp(argv[1], "both") || argc != 7) return 2;
  const int sq_bpsm = atoi(argv[2]), warps = atoi(argv[3]), bpsm = atoi(argv[4]), ch = atoi(argv[5]),
            reps = atoi(argv[6]);
  int sq_max = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sq_max, sq::chain<2>, 128, 0));
  cudaFuncAttributes sq_attr;
  CK(cudaFuncGetAttributes(&sq_attr, sq::chain<2>));
  const int n = sms * std::min(sq_bpsm, sq_max) * 128;
  std::vector<uint32_t> h;
  fill(h, n);
  uint32_t *xs;
  CK(cudaMalloc(&xs, h.size() * 4));
  CK(cudaMemcpy(xs, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  auto time_sq = [&](int K) {
    CK(cudaEventRecord(ts.a, s_sq));
    sq::chain<2><<<n / 128, 128, 0, s_sq>>>(xs, n, K);
    CK(cudaEventRecord(ts.b, s_sq));
    CK(cudaEventSynchronize(ts.b));
    float ms;
    CK(cudaEventElapsedTime(&ms, ts.a, ts.b));
    return ms;
  };
  const int grid = sms * bpsm, threads = 32 * warps;
  time_sq(100);
  const int K = std::max(1, (int)(1000.0 * 25.0 / time_sq(1000)));
  time_mma(ch, grid, threads, 100);
  const uint64_t iters = std::max<uint64_t>(100, (uint64_t)(2000.0 * 25.0 / time_mma(ch, grid, threads, 2000)));
  auto both = [&](bool sq_first, float &sq_end, float &mm_end) {
    CK(cudaEventRecord(t0, s_sq));
    CK(cudaStreamWaitEvent(s_mm, t0, 0));
    if (sq_first) {
      sq::chain<2><<<n / 128, 128, 0, s_sq>>>(xs, n, K);
      launch_ch(ch, grid, threads, iters, sink, s_mm);
    } else {
      launch_ch(ch, grid, threads, iters, sink, s_mm);
      sq::chain<2><<<n / 128, 128, 0, s_sq>>>(xs, n, K);
    }
    CK(cudaEventRecord(ts.b, s_sq));
    CK(cudaEventRecord(tm.b, s_mm));
    CK(cudaEventSynchronize(ts.b));
    CK(cudaEventSynchronize(tm.b));
    CK(cudaEventElapsedTime(&sq_end, t0, ts.b));
    CK(cudaEventElapsedTime(&mm_end, t0, tm.b));
  };
  std::vector<float> sq_a, mm_a, s1_sq, s1_mm, m1_sq, m1_mm;
  for (int r = 0; r < reps; ++r) {
    sq_a.push_back(time_sq(K));
    mm_a.push_back(time_mma(ch, grid, threads, iters));
    float a, b;
    both(true, a, b);
    s1_sq.push_back(a), s1_mm.push_back(b);
    both(false, a, b);
    m1_sq.push_back(a), m1_mm.push_back(b);
  }
  const double sq_work = (double)n * K, mm_work = (double)grid * warps * ch * iters;
  const float tsq = median(sq_a), tmm = median(mm_a);
  printf("{\"mode\": \"both\", \"bits\": %d, \"sms\": %d, \"sq_regs\": %d, \"sq_blocks_per_sm\": %d, \"warps\": %d, "
         "\"blocks_per_sm\": %d, \"chains\": %d, \"mma_regs\": %d, \"K\": %d, \"iters\": %llu, "
         "\"sq_alone_ms\": %.3f, \"sq_rate_alone\": %.4e, \"mma_alone_ms\": %.3f, \"mma_rate_alone\": %.4e, "
         "\"sq_first_sq_end_ms\": %.3f, \"sq_first_mma_end_ms\": %.3f, \"mma_first_sq_end_ms\": %.3f, "
         "\"mma_first_mma_end_ms\": %.3f, \"work_share_sum_sq_first\": %.3f, \"work_share_sum_mma_first\": %.3f}\n",
         SQ_BITS, sms, sq_attr.numRegs, std::min(sq_bpsm, sq_max), warps, bpsm, ch, regs_ch(ch), K,
         (unsigned long long)iters, tsq, sq_work / (tsq * 1e-3), tmm, mm_work / (tmm * 1e-3), median(s1_sq),
         median(s1_mm), median(m1_sq), median(m1_mm),
         (tsq + tmm) / std::max(median(s1_sq), median(s1_mm)), (tsq + tmm) / std::max(median(m1_sq), median(m1_mm)));
  return 0;
}
