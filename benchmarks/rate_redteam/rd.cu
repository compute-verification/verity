// The red team's read-and-rebuild probe (lane ra-rate-redteam). In every block, warps [0, rw) stream a device buffer
// far larger than L2 (the adversary's reads of the words it keeps) and the other warps run the word codec's k-round
// encode (its rebuild of the words it doesn't), all for the same number of SM clocks. F1 charges the rebuild only to
// the integer ALU the window's other work leaves free, so a read-bound pass leaves nearly all of it: at some split the
// read rate stays near its bare rate while the encode rate stays near its own, and the two shares sum to well above 1.
//
//   rd gate  K KEYFILE OUT        fixed steps on two blocks over a small buffer that wraps: codec chains to OUT for
//                                 gate.py, every read thread's XOR checked here against the host's
//   rd sweep K KEYFILE OUT [MiB]  rw = 0 … 8 at the occupancy limit, three reps each, JSON to OUT
//
// KEYFILE: gate.py's "k c_0 … c_{k-1} tau tstep", decimal.
#include "../word_codec/wc.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace rd {

constexpr int UE = 4;  // codec chains a thread, as co.cu
constexpr int VL = 8;  // 16-byte loads in flight a read thread
constexpr int THREADS = 256;

// SplitMix64's finalizer: the first fill (an arithmetic progression XOR a slow counter) read back at 2.5× the
// GDDR7 peak in the mixed splits, so the memory system compressed it.
__host__ __device__ inline uint32_t fill(uint64_t i) {
  uint64_t z = i * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return (uint32_t)(z ^ (z >> 31));
}

__global__ void fill_kernel(uint4* buf, size_t n16) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
    buf[i] = make_uint4(fill(4 * i), fill(4 * i + 1), fill(4 * i + 2), fill(4 * i + 3));
}

// cycles > 0: run for that many SM clocks; cycles < 0: exactly -cycles steps. counts[0] bytes read, counts[1] words.
// Needs VL * (read threads) <= n16, so each wrap is one subtraction.
template <int K>
__global__ void __launch_bounds__(THREADS) rd_kernel(int rw, long long cycles, uint64_t tstep,
                                                     const uint4* __restrict__ buf, size_t n16,
                                                     unsigned long long* counts, uint64_t* out_e, uint32_t* out_r) {
  const int warp = threadIdx.x >> 5;
  const uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const bool fixed = cycles < 0;
  const long long steps = -cycles;
  const long long t0 = clock64();
  unsigned long long n = 0;
  if (warp < rw) {
    const size_t rt = (size_t)gridDim.x * rw * 32;
    const size_t me = (size_t)blockIdx.x * rw * 32 + threadIdx.x;
    size_t off = me;
    uint32_t acc = 0;
    for (long long s = 0; fixed ? s < steps : clock64() - t0 < cycles; ++s, ++n) {
      uint4 v[VL];
#pragma unroll
      for (int j = 0; j < VL; ++j) {
        size_t o = off + j * rt;
        if (o >= n16) o -= n16;
        v[j] = __ldcs(buf + o);
      }
#pragma unroll
      for (int j = 0; j < VL; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
      off += VL * rt;
      if (off >= n16) off -= n16;
    }
    atomicAdd(&counts[0], n * VL * 16);
    if (out_r) out_r[me] = acc;
  } else {
    uint64_t y[UE], t[UE];
#pragma unroll
    for (int u = 0; u < UE; ++u) {
      y[u] = wc::canon((id * UE + u) * 0x9E3779B97F4A7C15ull & wc::P);
      t[u] = wc::mulmod(wc::kTau, id * UE + u);
    }
    for (long long s = 0; fixed ? s < steps : clock64() - t0 < cycles; ++s, ++n) {
#pragma unroll
      for (int u = 0; u < UE; ++u) {
        y[u] = wc::encode_word<K>(y[u], t[u]);
        t[u] = wc::addmod(t[u], tstep);
      }
    }
    atomicAdd(&counts[1], n * UE);
    if (out_e)
#pragma unroll
      for (int u = 0; u < UE; ++u) out_e[id * UE + u] = y[u];
  }
}

struct Run {
  double ms, bytes_per_s, words_per_s;
};

template <int K>
Run launch(int grid, int rw, long long cycles, uint64_t tstep, const uint4* buf, size_t n16, uint64_t* out_e,
           uint32_t* out_r) {
  unsigned long long* counts;
  cudaMalloc(&counts, 2 * sizeof(unsigned long long));
  cudaMemset(counts, 0, 2 * sizeof(unsigned long long));
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  rd_kernel<K><<<grid, THREADS>>>(rw, cycles, tstep, buf, n16, counts, out_e, out_r);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  unsigned long long c[2];
  cudaMemcpy(c, counts, sizeof c, cudaMemcpyDeviceToHost);
  cudaFree(counts);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return {ms, c[0] / (ms * 1e-3), c[1] / (ms * 1e-3)};
}

uint4* make_buf(size_t n16) {
  uint4* buf;
  if (cudaMalloc(&buf, n16 * sizeof(uint4))) return nullptr;
  fill_kernel<<<1024, 256>>>(buf, n16);
  cudaDeviceSynchronize();
  return buf;
}

template <int K>
int gate(uint64_t tstep, const char* out) {
  const int grid = 2, rw = 3, steps = 50;
  const size_t n16 = 1 << 16, n = (size_t)grid * THREADS, rt = (size_t)grid * rw * 32;
  uint4* buf = make_buf(n16);
  uint64_t* de;
  uint32_t* dr;
  cudaMalloc(&de, n * UE * sizeof(uint64_t));
  cudaMalloc(&dr, rt * sizeof(uint32_t));
  launch<K>(grid, rw, -steps, tstep, buf, n16, de, dr);
  std::vector<uint64_t> e(n * UE);
  std::vector<uint32_t> r(rt);
  cudaMemcpy(e.data(), de, e.size() * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(r.data(), dr, r.size() * 4, cudaMemcpyDeviceToHost);
  if (cudaGetLastError()) return 3;
  int bad = 0;
  for (size_t me = 0; me < rt; ++me) {
    uint32_t acc = 0;
    size_t off = me;
    for (int s = 0; s < steps; ++s, off = (off + VL * rt) % n16)
      for (int j = 0; j < VL; ++j) {
        const size_t o = (off + j * rt) % n16;
        for (int w = 0; w < 4; ++w) acc ^= fill(4 * o + w);
      }
    bad += acc != r[me];
  }
  FILE* f = fopen(out, "w");
  for (size_t id = 0; id < n; ++id)
    if ((id % THREADS) / 32 >= (size_t)rw)
      for (int u = 0; u < UE; ++u) fprintf(f, "%llu %llu\n", (unsigned long long)(id * UE + u), (unsigned long long)e[id * UE + u]);
  fprintf(f, "steps %d\n", steps);
  fclose(f);
  printf("gate: %zu read threads checked on the host (%d wraps each), %d differ\n", rt, (int)(steps * VL * rt / n16), bad);
  cudaFree(de);
  cudaFree(dr);
  cudaFree(buf);
  return bad ? 2 : 0;
}

template <int K>
int sweep(uint64_t tstep, const char* out, long long cycles, size_t mib) {
  int dev = 0, sms = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, rd_kernel<K>, THREADS, 0);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, rd_kernel<K>);
  const int grid = sms * per_sm;
  const size_t n16 = (mib << 20) / sizeof(uint4);
  uint4* buf = make_buf(n16);
  if (!buf) return 4;
  launch<K>(grid, 4, cycles / 10, tstep, buf, n16, nullptr, nullptr);  // warm-up
  FILE* f = fopen(out, "w");
  fprintf(f, "{\"k\": %d, \"sqr\": %d, \"sms\": %d, \"blocks_per_sm\": %d, \"regs\": %d, \"cycles\": %lld, \"ue\": %d, "
             "\"vl\": %d, \"buffer_mib\": %zu, \"rows\": [",
          K, WC_SQR, sms, per_sm, fa.numRegs, cycles, UE, VL, mib);
  for (int rw = 0; rw <= 8; ++rw) {
    std::vector<Run> r;
    for (int rep = 0; rep < 3; ++rep) r.push_back(launch<K>(grid, rw, cycles, tstep, buf, n16, nullptr, nullptr));
    std::sort(r.begin(), r.end(), [](const Run& x, const Run& y) { return x.ms < y.ms; });
    const Run& m = r[1];
    fprintf(f, "%s{\"rw\": %d, \"ms\": %.4f, \"bytes_per_s\": %.6e, \"words_per_s\": %.6e}", rw ? ", " : "", rw, m.ms,
            m.bytes_per_s, m.words_per_s);
    printf("rw %d: %.3f ms, %.4e B/s, %.4e words/s\n", rw, m.ms, m.bytes_per_s, m.words_per_s);
  }
  fprintf(f, "]}\n");
  fclose(f);
  cudaFree(buf);
  return cudaGetLastError() ? 3 : 0;
}

}  // namespace rd

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: rd gate|sweep K KEYFILE OUT [MiB] [CYCLES]\n");
    return 64;
  }
  const std::string mode = argv[1];
  const int k = atoi(argv[2]);
  std::ifstream kf(argv[3]);
  int kk = 0;
  kf >> kk;
  if (kk != k || k < 1 || k > 3) return 65;
  uint64_t c[wc::MAXK] = {}, tau = 0, tstep = 0;
  for (int j = 0; j < k; ++j) kf >> c[j];
  kf >> tau >> tstep;
  if (!kf || wc_set_key(c, k, tau)) return 66;
  const size_t mib = argc > 5 ? atoll(argv[5]) : 4096;
  const long long cycles = argc > 6 ? atoll(argv[6]) : 400000000LL;
#define RD_CASE(K) \
  if (k == K) return mode == "gate" ? rd::gate<K>(tstep, argv[4]) : rd::sweep<K>(tstep, argv[4], cycles, mib);
  RD_CASE(1) RD_CASE(2) RD_CASE(3)
#undef RD_CASE
  return 67;
}
