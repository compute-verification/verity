// M4e's E4M3 part: is tcgen05.mma kind::f8f6f4's FP32 accumulation exact in
// emu8's binades? The one question B300 (no kind::i8) hangs on; it runs on a B200, the same kind.
//
// emu8 writes an s8 product through E4M3 limb planes: x = 16 h + l,
// h = (x + 8) >> 4, l in [-8, 7], m = h + l. k3: acc1 = C1 + (4h)(4h')^T + (-l)(l')^T, acc2 = C2 + m m'^T, and the k32
// word is 15 bits(acc1) + 2 bits(acc2) mod 2^32 (C1 = 0x4B7BAB6C ulp 1, C2 = 0x49E07A56 ulp 1/8, 15 C1 + 2 C2 = 0).
// k4: acc = C0 + (16h)(16h')^T + (16h) l'^T + l (16h')^T + l l'^T, word = bits(acc) - C0 (C0 = 1.5 2^23).
// One CTA: TMEM preloaded with the constant (tcgen05.st), the arm's MMAs chained (128 x 256 x 32 each), the accumulators
// read back (tcgen05.ld). For each operand class: `float_bad`, accumulators whose bits differ from the exact C + T, and
// `word_bad`, k32 words that differ from the s8 dot product. Exact means both are 0 over every class.
//
//   fp8acc gate DIR     every arm x class; one JSON line each and DIR/summary.json; exit 0 exact, 4 not, 2 no device
//   fp8acc selftest DIR the same with ideal accumulators (no GPU): the planes, E4M3 encoding and word identity
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a fp8acc.cu -o fp8acc   (run.sh sm100)
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dc.cuh"

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

constexpr int BM = 128, BN = 256, KB = 32, MAXP = 4, NPL = 3;  // MMAs a job, planes a side
constexpr uint32_t C0 = 0x4B400000u, C1 = 0x4B7BAB6Cu, C2 = 0x49E07A56u;

struct Job {
  int nprod;              // MMAs chained
  int pa[MAXP], pb[MAXP]; // plane of A and of B for each MMA
  int acc[MAXP];          // accumulator (0: columns 0-255, 1: 256-511) of each MMA
  uint32_t c[2];          // each accumulator's starting bits
};

// NPL A planes (BM x KB E4M3 bytes each, row-major) and B planes (BN x KB) in global memory; D: 2 x BM x BN bits.
__global__ void __launch_bounds__(128) fp8acc(const uint8_t* A, const uint8_t* B, Job job, uint32_t* D) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  __shared__ __align__(1024) unsigned char sA[NPL][BM * KB];
  __shared__ __align__(1024) unsigned char sB[NPL][BN * KB];
  __shared__ uint64_t bar;
  __shared__ uint32_t tslot;
  const uint32_t w = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
  for (int p = 0; p < NPL; p++) {
    for (int c = threadIdx.x; c < BM * 2; c += blockDim.x)
      *reinterpret_cast<uint4*>(sA[p] + kmaj_off(c >> 1, 16 * (c & 1), KB)) =
          *reinterpret_cast<const uint4*>(A + (size_t)p * BM * KB + (c >> 1) * KB + 16 * (c & 1));
    for (int c = threadIdx.x; c < BN * 2; c += blockDim.x)
      *reinterpret_cast<uint4*>(sB[p] + kmaj_off(c >> 1, 16 * (c & 1), KB)) =
          *reinterpret_cast<const uint4*>(B + (size_t)p * BN * KB + (c >> 1) * KB + 16 * (c & 1));
  }
  if (threadIdx.x == 0) {
    mbar_init(&bar, 1);
    fence_mbar_init();
  }
  if (w == 0) tmem_alloc(&tslot, 512);
  fence_async_smem();
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = tslot, trow = tmem + ((32u * w) << 16);
  for (int a = 0; a < 2; a++) {
    uint32_t r[16];
#pragma unroll
    for (int i = 0; i < 16; i++) r[i] = job.c[a];
    for (int u = 0; u < BN / 16; u++) tmem_st16(trow + 256 * a + 16 * u, r);
  }
  tmem_wait_st();
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  constexpr uint32_t idesc = umma_idesc(1, 0, 0, BM, BN);
  if (w == 0) {
    if (elect_one()) {
      for (int i = 0; i < job.nprod; i++)
        umma_f8f6f4(tmem + 256u * job.acc[i], umma_desc(smem_u32(sA[job.pa[i]]), kmaj_lbo(), kmaj_sbo(KB)),
                    umma_desc(smem_u32(sB[job.pb[i]]), kmaj_lbo(), kmaj_sbo(KB)), idesc, 1);
      umma_commit(&bar);
    }
    __syncwarp();
  }
  mbar_wait(&bar, 0);
  tc_after_sync();
  const uint32_t row = 32 * w + lane;
  for (int a = 0; a < 2; a++)
    for (int u = 0; u < BN / 16; u++) {
      uint32_t r[16];
      tmem_ld16(trow + 256 * a + 16 * u, r);
      tmem_wait_ld();
      for (int i = 0; i < 16; i++) D[((size_t)a * BM + row) * BN + 16 * u + i] = r[i];
    }
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (w == 0) tmem_dealloc(tmem, 512);
#endif
}

// ---- host ----------------------------------------------------------------------------------------------------------------
// E4M3 of an integer it holds exactly (|v| <= 448, at most 4 significant bits).
static uint8_t e4m3(int v) {
  if (v == 0) return 0;
  const uint8_t s = v < 0 ? 0x80 : 0;
  int a = std::abs(v), p = 0;
  while ((a >> (p + 1)) != 0) p++;
  const int mant = p >= 3 ? (a >> (p - 3)) - 8 : (a << (3 - p)) - 8;
  if (p >= 3 && ((a >> (p - 3)) << (p - 3)) != a) { fprintf(stderr, "ERR %d is not an E4M3 integer\n", v); exit(2); }
  return s | (uint8_t)((p + 7) << 3) | (uint8_t)mant;
}
static double e4m3_value(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
  return (b & 0x80) ? -v : v;
}
static uint32_t mix(uint32_t x) { return dc_mix32(x); }
// Operand classes (emu8's adversaries, reduced): uniform s8, every product +max, every product -max (both -128),
// max magnitude with independent signs, l = -8 with the largest h.
static int8_t draw(int cls, uint32_t which, uint32_t i) {
  const uint32_t r = mix(mix(0xD8u + cls * 0x1000u + which) + i);
  switch (cls) {
    case 0: return (int8_t)(r & 0xFF);
    case 1: return 127;
    case 2: return -128;
    case 3: return (r & 1) ? 127 : -128;
    default: return (r & 1) ? 120 : -120;
  }
}
static const char* CLASSES[] = {"uniform", "max_pos", "min_min", "max_mag_signs", "l_min_h_max"};

// sim: no device; the accumulators are the exact C + T, so only the host's planes, encoding and word identity are tested.
static int run_gate(const std::string& dir, bool sim) {
  uint8_t *dA = nullptr, *dB = nullptr;
  uint32_t* dD = nullptr;
  if (!sim) {
    CK(cudaMalloc(&dA, NPL * BM * KB));
    CK(cudaMalloc(&dB, NPL * BN * KB));
    CK(cudaMalloc(&dD, 2 * BM * BN * 4));
  }
  long enc_bad = 0;
  std::vector<uint32_t> D(2 * BM * BN);
  int bad_total = 0;
  FILE* sum = fopen((dir + "/summary.json").c_str(), "w");
  if (!sum) { fprintf(stderr, "ERR open %s/summary.json\n", dir.c_str()); return 2; }
  fprintf(sum, "[");
  for (int arm = 0; arm < 2; arm++)    // 0: k3, 1: k4
    for (int cls = 0; cls < 5; cls++) {
      std::vector<int8_t> X(BM * KB), W(BN * KB);
      for (int i = 0; i < BM * KB; i++) X[i] = draw(cls, 1, i);
      for (int i = 0; i < BN * KB; i++) W[i] = draw(cls, 2, i);
      auto h = [](int x) { return (x + 8) >> 4; };
      // plane values: k3 A [4h, m, -l], B [4h', m', l']; k4 A [16h, l], B [16h', l']
      std::vector<int> av[NPL], bv[NPL];
      for (int p = 0; p < NPL; p++) { av[p].assign(BM * KB, 0); bv[p].assign(BN * KB, 0); }
      auto fill = [&](std::vector<int>* v, const std::vector<int8_t>& x, bool bside) {
        for (size_t i = 0; i < x.size(); i++) {
          const int hh = h(x[i]), l = x[i] - 16 * hh, m = hh + l;
          if (arm == 0) { v[0][i] = 4 * hh; v[1][i] = m; v[2][i] = bside ? l : -l; }
          else { v[0][i] = 16 * hh; v[1][i] = l; }
        }
      };
      fill(av, X, false);
      fill(bv, W, true);
      Job job{};
      if (arm == 0) job = Job{3, {0, 2, 1}, {0, 2, 1}, {0, 0, 1}, {C1, C2}};
      else job = Job{4, {0, 0, 1, 1}, {0, 1, 0, 1}, {0, 0, 0, 0}, {C0, C0}};
      std::vector<uint8_t> A(NPL * BM * KB), B(NPL * BN * KB);
      for (int p = 0; p < NPL; p++) {
        for (int i = 0; i < BM * KB; i++) A[p * BM * KB + i] = e4m3(av[p][i]);
        for (int i = 0; i < BN * KB; i++) B[p * BN * KB + i] = e4m3(bv[p][i]);
        for (int i = 0; i < BM * KB; i++) enc_bad += e4m3_value(A[p * BM * KB + i]) != av[p][i];
        for (int i = 0; i < BN * KB; i++) enc_bad += e4m3_value(B[p * BN * KB + i]) != bv[p][i];
      }
      if (!sim) {
        CK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice));
        CK(cudaMemset(dD, 0xA5, D.size() * 4));
        fp8acc<<<1, 128>>>(dA, dB, job, dD);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost));
      }
      long float_bad = 0, word_bad = 0;
      double worst = 0;
      for (int i = 0; i < BM; i++)
        for (int j = 0; j < BN; j++) {
          double T[2] = {0, 0};
          for (int q = 0; q < job.nprod; q++) {
            long s = 0;
            for (int k = 0; k < KB; k++) s += (long)av[job.pa[q]][i * KB + k] * bv[job.pb[q]][j * KB + k];
            T[job.acc[q]] += (double)s;
          }
          uint32_t got[2];
          for (int a = 0; a < (arm == 0 ? 2 : 1); a++) {
            float c;
            std::memcpy(&c, &job.c[a], 4);
            const double want = (double)c + T[a];
            const float wf = (float)want;
            uint32_t wb;
            std::memcpy(&wb, &wf, 4);
            got[a] = sim ? wb : D[((size_t)a * BM + i) * BN + j];
            float gf;
            std::memcpy(&gf, &got[a], 4);
            if (got[a] != wb || (double)wf != want) {
              float_bad++;
              worst = std::fmax(worst, std::fabs((double)gf - want));
            }
          }
          int32_t dot = 0;
          for (int k = 0; k < KB; k++) dot += (int32_t)X[i * KB + k] * W[j * KB + k];
          const uint32_t word = arm == 0 ? 15u * got[0] + 2u * got[1] : got[0] - C0;
          word_bad += word != (uint32_t)dot;
        }
      bad_total += float_bad + word_bad;
      char line[320];
      snprintf(line, sizeof line,
               "{\"arm\": \"%s\", \"class\": \"%s\", \"mmas\": %d, \"accumulators\": %d, \"float_bad\": %ld, "
               "\"word_bad\": %ld, \"worst_abs_err\": %.17g}",
               arm == 0 ? "k3" : "k4", CLASSES[cls], job.nprod, arm == 0 ? 2 * BM * BN : BM * BN, float_bad, word_bad, worst);
      printf("%s\n", line);
      fprintf(sum, "%s%s", arm + cls ? ",\n " : "", line);
    }
  fprintf(sum, "]\n");
  fclose(sum);
  bad_total += enc_bad;
  printf("{\"mode\": \"%s\", \"encoding_bad\": %ld, \"exact\": %s}\n", sim ? "selftest" : "fp8acc", enc_bad,
         bad_total ? "false" : "true");
  return bad_total ? 4 : 0;
}

int main(int argc, char** argv) {
  const std::string mode = argc == 3 ? argv[1] : "";
  if (mode != "gate" && mode != "selftest") {
    fprintf(stderr, "usage: fp8acc gate|selftest DIR\n");
    return 2;
  }
  if (mode == "selftest") return run_gate(argv[2], true);
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  if (p.major != 10) { fprintf(stderr, "ERR device sm_%d%d: kind::f8f6f4 here needs sm_100a\n", p.major, p.minor); return 2; }
  return run_gate(argv[2], false);
}
