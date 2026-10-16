// HC-PREFILL's CUDA-core probe (cmp-13, approach pouw/filler-prefill): how much int8 multiply-accumulate the CUDA cores do
// beside the scorer's prefill linears. ncu reads the prefill GEMM (cutlass_80_tensorop_i16832gemm_s8_128x128_64x6_tn) at
// 16% of issue slots busy with its tensor pipe at 84% (r20261016-071430-ef97), and the scorer's capacity C counts only the
// tensor pipe (1024 int8 MACs an SM a clock). This measures dp4a (4 int8 MACs a lane an instruction, NC dependent chains
// a thread) on the CUDA cores: alone at full occupancy and at a co shape, then pumped on the lowest-priority stream
// beside the 32 layers of linears at M = 8192 on the highest (dscore.cu's prefill workload, ABBA over ROUNDS rounds).
// No receipts: it is a capacity measurement, not a filler.
//
//   hcpre_dp4a ROUNDS K BLOCK NC ITERS     one JSON line: alone rates, then the GEMMs alone vs beside the pump
#define main dcap_main
#include "../decode_capacity/dcap.cu"
#undef main

constexpr size_t MAX_BL = 786432;

// dscore.cu's setup, as it is there (dscore.cu defines main, so it can't be included)
static int setup(Dev& d, bool weights, size_t kv_want) {
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&d.st));
  CK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, 0));
  if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); return 2; }
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
  if (nvmlDeviceGetHandleByPciBusId_v2(bus, &d.nv) != NVML_SUCCESS) { fprintf(stderr, "NVML handle failed\n"); return 2; }
  char uuid[96] = {0}, name[96] = {0};
  nvmlDeviceGetUUID(d.nv, uuid, sizeof uuid);
  nvmlDeviceGetName(d.nv, name, sizeof name);
  const char* want = getenv("GPU_LEASE_UUID");
  if (want && *want && strcmp(want, uuid) != 0) {
    fprintf(stderr, "device %s is not the leased %s\n", uuid, want);
    return 3;
  }
  fprintf(stderr, "device %s %s, %d SMs\n", name, uuid, d.sms);
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  CK(cudaMalloc(&g_ref, (size_t)256 * VOCAB * sizeof(int)));
  CK(cudaMalloc(&g_cnt, sizeof *g_cnt));
  if (weights) {
    for (int l = 0; l < LAYERS; l++)
      for (int j = 0; j < 4; j++) {
        const size_t n = (size_t)LN[j] * LK[j];
        CK(cudaMalloc(&d.W[l][j], n));
        fill_s8<<<1024, 256, 0, d.st>>>(d.W[l][j], n, 0x1000 + l * 4 + j);
      }
    CK(cudaMalloc(&d.Wlm, (size_t)VOCAB * HID));
    fill_s8<<<1024, 256, 0, d.st>>>(d.Wlm, (size_t)VOCAB * HID, 0x77);
  }
  CK(cudaMalloc(&d.X, (size_t)8192 * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(d.X, (size_t)8192 * FF, 0x55);
  CK(cudaMalloc(&d.D, (size_t)1 << 30));
  size_t fr, tot;
  CK(cudaMemGetInfo(&fr, &tot));
  d.kv_bytes = std::min({(size_t)48 << 30, fr - ((size_t)3 << 30), kv_want});
  CK(cudaMalloc(&d.kv, d.kv_bytes));
  fill_s8<<<4096, 256, 0, d.st>>>(d.kv, d.kv_bytes, 0x99);
  CK(cudaMalloc(&d.q, (size_t)256 * HQ * HD));
  fill_s8<<<64, 256, 0, d.st>>>(d.q, (size_t)256 * HQ * HD, 0x31);
  CK(cudaMalloc(&d.P, MAX_BL * HQ));
  fill_s8<<<1024, 256, 0, d.st>>>(d.P, MAX_BL * HQ, 0x32);
  CK(cudaMalloc(&d.S, MAX_BL * HQ * sizeof(int)));
  CK(cudaMalloc(&d.O, (size_t)256 * HQ * HD * sizeof(int)));
  CK(cudaMalloc(&d.sink, 64));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  return 0;
}


template <int NC>
__global__ void dp4a_k(uint32_t seed, int iters, unsigned long long* out) {
  int acc[NC];
  const uint32_t b = seed * 0x9E3779B9u + threadIdx.x;
#pragma unroll
  for (int c = 0; c < NC; c++) acc[c] = (int)(seed ^ (c * 0x85EBCA6Bu) ^ (blockIdx.x << 10) ^ threadIdx.x);
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int c = 0; c < NC; c++) acc[c] = __dp4a(acc[c], (int)b, acc[c]);
  }
  unsigned long long s = 0;
#pragma unroll
  for (int c = 0; c < NC; c++) s += (unsigned)acc[c];
  if (s == 0x1234567ull) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

struct DP {
  int grid, block, nc, iters;
  unsigned long long* out;
  double macs() const { return 4.0 * grid * block * nc * (double)iters; }
};
static void dp_launch(const DP& p, cudaStream_t s, uint32_t seed) {
  switch (p.nc) {
    case 4: dp4a_k<4><<<p.grid, p.block, 0, s>>>(seed, p.iters, p.out); break;
    case 8: dp4a_k<8><<<p.grid, p.block, 0, s>>>(seed, p.iters, p.out); break;
    case 16: dp4a_k<16><<<p.grid, p.block, 0, s>>>(seed, p.iters, p.out); break;
    default: fprintf(stderr, "NC 4, 8 or 16\n"); exit(1);
  }
}
// n launches back to back on s: MACs/s and the mean SM clock read while they ran
static double dp_alone(Dev& d, const DP& p, cudaStream_t s, int n, double* mhz) {
  dp_launch(p, s, 1);
  CK(cudaStreamSynchronize(s));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0, s));
  for (int i = 0; i < n; i++) dp_launch(p, s, 2 + i);
  CK(cudaEventRecord(e1, s));
  double cs = 0;
  int nc = 0;
  while (cudaEventQuery(e1) == cudaErrorNotReady) {
    const double c = nvml_mhz(d);
    if (c > 0) { cs += c; nc++; }
  }
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  *mhz = nc ? cs / nc : nvml_mhz(d);
  return n * p.macs() / (ms * 1e-3);
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "hcpre_dp4a ROUNDS K BLOCK NC ITERS\n"); return 1; }
  const int rounds = atoi(argv[1]), K = atoi(argv[2]), block = atoi(argv[3]), nc = atoi(argv[4]), iters = atoi(argv[5]);
  Dev d;
  if (int rc = setup(d, true, (size_t)1 << 20)) return rc;
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t full, fs;
  CK(cudaStreamCreateWithPriority(&full, cudaStreamDefault, hi));
  CK(cudaStreamCreateWithPriority(&fs, cudaStreamNonBlocking, lo));
  DP big{d.sms, 1024, 4, iters, nullptr}, co{d.sms, block, nc, iters, nullptr};
  CK(cudaMalloc(&big.out, (size_t)d.sms * 1024 * sizeof(unsigned long long)));
  co.out = big.out;
  double mb = 0, mc = 0;
  const double rb = dp_alone(d, big, fs, 20, &mb), rc_ = dp_alone(d, co, fs, 20, &mc);
  const std::vector<Lt> p = lin_plans(d, 8192);
  bool wrapped = false;
  d.st = full;
  auto work = [&] { enqueue_step(d, p, 0, 8192, 0, &wrapped); };
  std::vector<std::vector<Rep>> al(rounds), ar(rounds);
  std::vector<double> rate(rounds);
  for (int r = 0; r < rounds; r++) {
    auto arm = [&] {
      std::atomic<bool> on{true};
      std::atomic<long long> done{0};
      std::vector<double> t;
      std::thread th([&] {
        cudaEvent_t ev[3];
        for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync));
        for (long long k = 0; on.load(); k++) {
          if (k >= 3) {
            CK(cudaEventSynchronize(ev[k % 3]));
            t.push_back(now_ms());
            done++;
          }
          dp_launch(co, fs, (uint32_t)(100 + k));
          CK(cudaEventRecord(ev[k % 3], fs));
        }
        CK(cudaStreamSynchronize(fs));
        for (auto& e : ev) CK(cudaEventDestroy(e));
      });
      while (done.load() < 2) {
      }
      double t0 = 0, t1 = 0;
      ar[r] = time_work(d, work, K, &t0, &t1);
      on = false;
      th.join();
      long long n = 0;
      double first = 0, last = 0;
      for (double x : t)
        if (x > t0 && x <= t1) {
          if (!n++) first = x;
          last = x;
        }
      rate[r] = n > 1 ? (n - 1) * co.macs() / ((last - first) * 1e-3) : 0;
    };
    if (r % 2 == 0) {
      al[r] = time_work(d, work, K);
      arm();
    } else {
      arm();
      al[r] = time_work(d, work, K);
    }
  }
  printf("{\"mode\": \"dp4a\", \"sms\": %d, \"alone_full\": {\"block\": 1024, \"nc\": 4, \"mac_per_s\": %.6e, \"mhz\": %.1f}, "
         "\"alone_co\": {\"block\": %d, \"nc\": %d, \"mac_per_s\": %.6e, \"mhz\": %.1f}, \"iters\": %d, \"rounds\": [",
         d.sms, rb, mb, block, nc, rc_, mc, iters);
  for (int r = 0; r < rounds; r++) {
    printf("%s{\"alone\": [", r ? ", " : "");
    for (size_t i = 0; i < al[r].size(); i++) printf("%s[%.4f, %.0f]", i ? ", " : "", al[r][i].ms, al[r][i].mhz);
    printf("], \"arm\": [");
    for (size_t i = 0; i < ar[r].size(); i++) printf("%s[%.4f, %.0f]", i ? ", " : "", ar[r][i].ms, ar[r][i].mhz);
    printf("], \"dp4a_mac_per_s\": %.6e}", rate[r]);
  }
  printf("]}\n");
  CK(cudaFree(big.out));
  return 0;
}
