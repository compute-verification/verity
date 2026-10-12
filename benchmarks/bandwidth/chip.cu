// bwchip: how much an RTX PRO 6000's on-chip stores hold, measured, for S_chip (the bits a device holds on chip when σ
// arrives).  Each sweep finds where a store stops holding a working set: the knee is its capacity, and the first size
// that runs at the next level's speed bounds it from above (a store that held that much would serve it).
//
//   icache    straight-line code of N steps (each 4 independent add-then-xor pairs, a distinct 32-bit immediate in every
//             xor: 8 instructions, 128 bytes), looped; cycles per instruction against the code's bytes, for one warp
//             and for four (one a scheduler), on one SM.  Knees: the L0 instruction cache (per scheduler), the SM's L1
//             instruction cache, any level between it and L2.  The kernels are PTX that bwchip writes and ptxas
//             assembles (run_chip.sh, which checks each cubin's SASS holds the 8·N steps).
//   const     a pointer chase through __constant__ memory (64 KiB, the bank's size), one thread: cycles per load against
//             the footprint, at strides of 64 and 256 bytes (the line size from the ratio of the knees).
//   tlb       a pointer chase over N pages of a 16 GiB allocation, one load a page (each in its own L2 set, L1
//             bypassed, all held by L2), at 64 KiB and 2 MiB strides: cycles per load against N; the knees are the
//             TLB levels' entries.
//   l1        a pointer chase through F bytes on one SM, cached in L1 (no shared memory carved out): the L1's capacity.
//   l2        a random pointer chase over F bytes by 4,096 chasers in 128-byte lines: the mean latency against F, from
//             all hits to all misses, and the bytes L2 retained h(F)·F with h from the latencies at 16 MiB and 2 GiB.
//
// Every kernel's result is checked against the host's (the chain's final state, the chase's final index) before its
// time is reported; a mismatch exits 3.  Writes --out/chip.json.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define OK(x)                                                                                           \
  do {                                                                                                  \
    cudaError_t e_ = (x);                                                                               \
    if (e_ != cudaSuccess) {                                                                            \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(4);                                                                                          \
    }                                                                                                   \
  } while (0)

namespace chip {

// ---------------------------------------------------------------------------------------------------------------------
// The instruction cache: code of N steps.

__host__ __device__ constexpr uint32_t imm(uint32_t i) {
  i *= 0x9E3779B1u;
  i ^= i >> 15;
  i *= 0x85EBCA77u;
  i ^= i >> 13;
  return i & 0x7fffffffu;
}

// nvcc takes minutes over straight-line code this long, so bwchip writes each size's kernel as PTX (--emit-icache DIR,
// icache_N.ptx), run_chip.sh assembles it with ptxas (seconds) and records its SASS, and the sweep loads the cubins.
static const int ICACHE_STEPS[] = {16,  32,  48,  64,   96,   128,  192,  256,  320,  384,  448,
                                   512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192};

// icache(out, cycles, reps, y): reps passes over the n steps (pass 0 cold and untimed); thread t writes its four chains
// to out[4t..4t+3] and the timed passes' cycles to cycles[t].
static std::string icache_ptx(int n) {
  std::string s =
      ".version 8.7\n.target sm_120\n.address_size 64\n\n"
      ".visible .entry icache(.param .u64 out, .param .u64 cyc, .param .u32 reps, .param .u32 y)\n{\n"
      " .reg .b32 %a, %b, %c, %d, %y, %r, %n, %i;\n .reg .b64 %t0, %t1, %o, %q, %w;\n .reg .pred %p, %z;\n"
      " ld.param.u64 %o, [out];\n ld.param.u64 %q, [cyc];\n ld.param.u32 %n, [reps];\n ld.param.u32 %y, [y];\n"
      " cvta.to.global.u64 %o, %o;\n cvta.to.global.u64 %q, %q;\n mov.u32 %i, %tid.x;\n"
      " mov.u32 %a, %i;\n add.u32 %b, %i, 1;\n add.u32 %c, %i, 2;\n add.u32 %d, %i, 3;\n"
      " mov.u32 %r, 0;\n mov.u64 %t0, 0;\nLOOP:\n setp.eq.u32 %z, %r, 1;\n @%z mov.u64 %t0, %clock64;\n";
  s.reserve(s.size() + (size_t)n * 4 * 64);
  char line[96];
  const char* reg[4] = {"%a", "%b", "%c", "%d"};
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 4; ++k) {
      snprintf(line, sizeof line, " add.u32 %s, %s, %%y;\n xor.b32 %s, %s, %u;\n", reg[k], reg[k], reg[k], reg[k],
               imm(4 * i + k));
      s += line;
    }
  s += " add.u32 %r, %r, 1;\n setp.lt.u32 %p, %r, %n;\n @%p bra LOOP;\n mov.u64 %t1, %clock64;\n"
       " sub.u64 %t1, %t1, %t0;\n mul.wide.u32 %w, %i, 16;\n add.u64 %w, %w, %o;\n"
       " st.global.v4.u32 [%w], {%a, %b, %c, %d};\n mul.wide.u32 %w, %i, 8;\n add.u64 %w, %w, %q;\n"
       " st.global.u64 [%w], %t1;\n ret;\n}\n";
  return s;
}

static void icache_host(uint32_t tid, int n, int reps, uint32_t y, uint32_t x[4]) {
  x[0] = tid, x[1] = tid + 1, x[2] = tid + 2, x[3] = tid + 3;
  for (int r = 0; r < reps; ++r)
    for (int s = 0; s < n; ++s)
      for (int k = 0; k < 4; ++k) x[k] = (x[k] + y) ^ imm(4 * s + k);
}

// ---------------------------------------------------------------------------------------------------------------------
// Pointer chases.

constexpr int CONST_WORDS = 16384;
__constant__ uint32_t cbank[CONST_WORDS];

__global__ void const_chase(uint32_t start, int warm, int n, uint32_t* out, unsigned long long* cycles) {
  uint32_t i = start;
#pragma unroll 1
  for (int k = 0; k < warm; ++k) i = cbank[i];
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int k = 0; k < n; ++k) i = cbank[i];
  const unsigned long long t1 = clock64();
  out[0] = i;
  cycles[0] = t1 - t0;
}

// One thread chasing 32-bit indices (in 4-byte words) through `p`, through L2 only (cg) or L1 too (ca).
template <bool L1>
__global__ void global_chase(const uint32_t* __restrict__ p, uint64_t start, int warm, int n, uint64_t* out,
                             unsigned long long* cycles) {
  uint64_t i = start;
#pragma unroll 1
  for (int k = 0; k < warm; ++k) i = L1 ? __ldca(p + i) : __ldcg(p + i);
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int k = 0; k < n; ++k) i = L1 ? __ldca(p + i) : __ldcg(p + i);
  const unsigned long long t1 = clock64();
  out[0] = i;
  cycles[0] = t1 - t0;
}

// Many chasers, one a warp, over one random cycle of line indices (each line's first word holds the next line).
__global__ void l2_chase(const uint32_t* __restrict__ p, const uint32_t* __restrict__ starts, int n, uint32_t* out,
                         unsigned long long* cycles) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (threadIdx.x % 32) return;
  uint32_t i = starts[w];
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int k = 0; k < n; ++k) i = __ldcg(p + (uint64_t)i * 32);
  const unsigned long long t1 = clock64();
  out[w] = i;
  cycles[w] = t1 - t0;
}

// ---------------------------------------------------------------------------------------------------------------------
// Host.

static std::string num(double x) {
  char b[64];
  snprintf(b, sizeof b, "%.6g", x);
  return b;
}

static int fail(const std::string& what) {
  fprintf(stderr, "GATE FAILED: %s\n", what.c_str());
  printf("GATE FAILED: %s\n", what.c_str());
  return 3;
}

struct Dev {
  int sms = 0, clock_khz = 0;
  size_t l2 = 0;
};

static int emit_icache(const std::string& dir) {
  for (int n : ICACHE_STEPS) {
    const std::string path = dir + "/icache_" + std::to_string(n) + ".ptx";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return fail("cannot write " + path);
    const std::string s = icache_ptx(n);
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
  }
  return 0;
}

static int run_icache(std::string& j, const std::string& dir) {
  uint32_t* out;
  unsigned long long* cyc;
  OK(cudaMalloc(&out, 128 * 16));
  OK(cudaMalloc(&cyc, 128 * 8));
  const uint32_t y = 0x5bd1e995u;
  j += "\"icache\":{\"bytes_per_step\":128,\"instructions_per_step\":8,\"runs\":[";
  bool first = true;
  for (int n : ICACHE_STEPS) {
    const std::string path = dir + "/icache_" + std::to_string(n) + ".cubin";
    cudaLibrary_t lib;
    cudaKernel_t fn;
    OK(cudaLibraryLoadFromFile(&lib, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    OK(cudaLibraryGetKernel(&fn, lib, "icache"));
    const int reps = std::max(4, (1 << 21) / (8 * n));
    for (int warps : {1, 4}) {
      std::vector<double> cpi;
      for (int t = 0; t < 3; ++t) {
        int r = reps;
        uint32_t yy = y;
        void* args[] = {&out, &cyc, &r, &yy};
        OK(cudaLaunchKernel((const void*)fn, dim3(1), dim3(32 * warps), args, 0, 0));
        OK(cudaGetLastError());
        OK(cudaDeviceSynchronize());
        std::vector<uint32_t> got(4 * 32 * warps);
        std::vector<unsigned long long> c(32 * warps);
        OK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(c.data(), cyc, c.size() * 8, cudaMemcpyDeviceToHost));
        for (uint32_t tid = 0; tid < 32u * warps; tid += 7) {
          uint32_t x[4];
          icache_host(tid, n, reps, y, x);
          if (memcmp(x, &got[4 * tid], 16)) return fail("icache result, n=" + std::to_string(n));
        }
        double worst = 0;
        for (auto x : c) worst = std::max(worst, (double)x);
        cpi.push_back(worst / ((double)(reps - 1) * 8 * n));
      }
      std::sort(cpi.begin(), cpi.end());
      j += std::string(first ? "" : ",") + "{\"steps\":" + std::to_string(n) + ",\"code_bytes\":" +
           std::to_string(128 * n) + ",\"warps\":" + std::to_string(warps) + ",\"reps\":" + std::to_string(reps) +
           ",\"cycles_per_instruction\":" + num(cpi[1]) + ",\"min\":" + num(cpi[0]) + ",\"max\":" + num(cpi[2]) + "}";
      first = false;
    }
    OK(cudaLibraryUnload(lib));
  }
  j += "]},";
  cudaFree(out);
  cudaFree(cyc);
  return 0;
}

static int run_const(std::string& j) {
  uint32_t* out;
  unsigned long long* cyc;
  OK(cudaMalloc(&out, 4));
  OK(cudaMalloc(&cyc, 8));
  j += "\"const\":{\"bank_bytes\":" + std::to_string(CONST_WORDS * 4) + ",\"runs\":[";
  bool first = true;
  for (int stride : {64, 256})
    for (int fp : {256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 49152,
                   57344, 65536}) {
      const int count = fp / stride, sw = stride / 4;
      if (count < 2) continue;
      std::vector<uint32_t> bank(CONST_WORDS, 0);
      for (int k = 0; k < count; ++k) bank[k * sw] = ((k + 1) % count) * sw;
      OK(cudaMemcpyToSymbol(cbank, bank.data(), bank.size() * 4));
      const int warm = 2 * count, n = std::max(4096, 8 * count);
      const_chase<<<1, 1>>>(0, warm, n, out, cyc);
      OK(cudaGetLastError());
      OK(cudaDeviceSynchronize());
      std::vector<double> lat;
      for (int t = 0; t < 3; ++t) {
        const_chase<<<1, 1>>>(0, warm, n, out, cyc);
        OK(cudaDeviceSynchronize());
        uint32_t got;
        unsigned long long c;
        OK(cudaMemcpy(&got, out, 4, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
        if (got != (uint32_t)(((warm + n) % count) * sw)) return fail("const chase, footprint " + std::to_string(fp));
        lat.push_back((double)c / n);
      }
      std::sort(lat.begin(), lat.end());
      j += std::string(first ? "" : ",") + "{\"stride\":" + std::to_string(stride) + ",\"footprint\":" +
           std::to_string(fp) + ",\"cycles_per_load\":" + num(lat[1]) + "}";
      first = false;
    }
  j += "]},";
  cudaFree(out);
  cudaFree(cyc);
  return 0;
}

// A single-thread chase over `count` elements at word offsets offs[k]: the median cycles a load over three timed runs.
template <bool L1>
static int chase(const uint32_t* dev, uint32_t* host_view_unused, const std::vector<uint64_t>& offs, double* out_lat,
                 uint64_t* scratch, unsigned long long* cyc, const std::string& what) {
  (void)host_view_unused;
  const int count = (int)offs.size();
  const int warm = 2 * count, n = std::max(4096, 4 * count);
  std::vector<double> lat;
  for (int t = 0; t < 4; ++t) {
    global_chase<L1><<<1, 1>>>(dev, offs[0], warm, n, scratch, cyc);
    OK(cudaGetLastError());
    OK(cudaDeviceSynchronize());
    uint64_t got;
    unsigned long long c;
    OK(cudaMemcpy(&got, scratch, 8, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
    if (got != offs[(warm + n) % count]) return fail(what);
    if (t) lat.push_back((double)c / n);
  }
  std::sort(lat.begin(), lat.end());
  *out_lat = lat[1];
  return 0;
}

// Writes the chain offs[k] -> offs[k+1] into device memory word by word.
static void link_chain(uint32_t* dev, const std::vector<uint64_t>& offs) {
  for (size_t k = 0; k < offs.size(); ++k) {
    const uint32_t next = (uint32_t)offs[(k + 1) % offs.size()];
    OK(cudaMemcpy(dev + offs[k], &next, 4, cudaMemcpyHostToDevice));
  }
}

static int run_tlb(std::string& j) {
  const size_t span = 16ull << 30;
  uint32_t* buf;
  OK(cudaMalloc(&buf, span));
  uint64_t* scratch;
  unsigned long long* cyc;
  OK(cudaMalloc(&scratch, 8));
  OK(cudaMalloc(&cyc, 8));
  j += "\"tlb\":{\"span\":" + std::to_string(span) + ",\"runs\":[";
  bool first = true;
  for (size_t page : {64ull << 10, 2ull << 20})
    for (int count : {4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
                      6144, 8192}) {
      if ((uint64_t)count * page > span) break;
      // Element k at page k, line (k mod 512) of it: one load a page, each in its own L2 set, 4 KiB to 1 MiB in all.
      std::vector<uint64_t> offs(count);
      for (int k = 0; k < count; ++k) offs[k] = ((uint64_t)k * page + (uint64_t)(k % 512) * 128 % page) / 4;
      if (offs.back() > UINT32_MAX) break;
      link_chain(buf, offs);
      double lat;
      if (int rc = chase<false>(buf, nullptr, offs, &lat, scratch, cyc, "tlb chase")) return rc;
      j += std::string(first ? "" : ",") + "{\"page\":" + std::to_string(page) + ",\"pages\":" + std::to_string(count) +
           ",\"cycles_per_load\":" + num(lat) + "}";
      first = false;
    }
  j += "]},";
  cudaFree(buf);
  cudaFree(scratch);
  cudaFree(cyc);
  return 0;
}

static int run_l1(std::string& j) {
  OK(cudaFuncSetAttribute(global_chase<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  const size_t span = 1 << 20;
  uint32_t* buf;
  OK(cudaMalloc(&buf, span));
  uint64_t* scratch;
  unsigned long long* cyc;
  OK(cudaMalloc(&scratch, 8));
  OK(cudaMalloc(&cyc, 8));
  j += "\"l1\":{\"line\":128,\"runs\":[";
  bool first = true;
  for (int kib : {8, 16, 32, 48, 64, 80, 96, 104, 112, 120, 128, 136, 144, 160, 192, 256}) {
    const int count = kib * 1024 / 128;
    std::vector<uint64_t> offs(count);
    for (int k = 0; k < count; ++k) offs[k] = (uint64_t)k * 32;
    link_chain(buf, offs);
    double lat;
    if (int rc = chase<true>(buf, nullptr, offs, &lat, scratch, cyc, "l1 chase")) return rc;
    j += std::string(first ? "" : ",") + "{\"footprint\":" + std::to_string(kib * 1024) + ",\"cycles_per_load\":" +
         num(lat) + "}";
    first = false;
  }
  j += "]},";
  cudaFree(buf);
  cudaFree(scratch);
  cudaFree(cyc);
  return 0;
}

static int run_l2(std::string& j, const Dev& d) {
  const size_t max_bytes = 2ull << 30;
  uint32_t* buf;
  OK(cudaMalloc(&buf, max_bytes));
  const int chasers = 4096, threads = 256;
  uint32_t *starts, *out;
  unsigned long long* cyc;
  OK(cudaMalloc(&starts, chasers * 4));
  OK(cudaMalloc(&out, chasers * 4));
  OK(cudaMalloc(&cyc, chasers * 8));
  std::mt19937_64 rng(12345);
  j += "\"l2\":{\"line\":128,\"chasers\":" + std::to_string(chasers) + ",\"runs\":[";
  std::vector<std::pair<size_t, double>> lat;
  for (size_t mib : {16, 32, 64, 96, 112, 120, 128, 136, 144, 160, 192, 256, 512, 2048}) {
    const size_t bytes = mib << 20, lines = bytes / 128;
    std::vector<uint32_t> order(lines);
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<uint32_t> img(bytes / 4, 0);
    for (size_t k = 0; k < lines; ++k) img[(size_t)order[k] * 32] = order[(k + 1) % lines];
    OK(cudaMemcpy(buf, img.data(), bytes, cudaMemcpyHostToDevice));
    std::vector<uint32_t> st(chasers);
    const size_t per = lines / chasers;
    for (int w = 0; w < chasers; ++w) st[w] = order[(size_t)w * per];
    OK(cudaMemcpy(starts, st.data(), chasers * 4, cudaMemcpyHostToDevice));
    std::vector<double> runs;
    for (int t = 0; t < 3; ++t) {  // pass 0 warms (each chaser covers its share of the cycle once a pass)
      l2_chase<<<chasers * 32 / threads, threads>>>(buf, starts, (int)per, out, cyc);
      OK(cudaGetLastError());
      OK(cudaDeviceSynchronize());
      std::vector<uint32_t> got(chasers);
      std::vector<unsigned long long> c(chasers);
      OK(cudaMemcpy(got.data(), out, chasers * 4, cudaMemcpyDeviceToHost));
      OK(cudaMemcpy(c.data(), cyc, chasers * 8, cudaMemcpyDeviceToHost));
      for (int w = 0; w < chasers; w += 97)
        if (got[w] != order[((size_t)w * per + per) % lines]) return fail("l2 chase, " + std::to_string(mib) + " MiB");
      double sum = 0;
      for (auto x : c) sum += (double)x;
      if (t) runs.push_back(sum / chasers / per);
    }
    lat.push_back({bytes, 0.5 * (runs[0] + runs[1])});
  }
  const double hit = lat.front().second, miss = lat.back().second;
  bool first = true;
  for (auto& [bytes, l] : lat) {
    const double h = std::min(1.0, std::max(0.0, (miss - l) / (miss - hit)));
    j += std::string(first ? "" : ",") + "{\"footprint\":" + std::to_string(bytes) + ",\"cycles_per_load\":" + num(l) +
         ",\"hit_fraction\":" + num(h) + ",\"retained_bytes\":" + num(h * bytes) + "}";
    first = false;
  }
  j += "],\"l2_bytes_attribute\":" + std::to_string(d.l2) + "},";
  cudaFree(buf);
  cudaFree(starts);
  cudaFree(out);
  cudaFree(cyc);
  return 0;
}

}  // namespace chip

int main(int argc, char** argv) {
  std::string out = ".", only, icache_dir, emit;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--out" && i + 1 < argc) out = argv[++i];
    else if (k == "--only" && i + 1 < argc) only = argv[++i];
    else if (k == "--icache" && i + 1 < argc) icache_dir = argv[++i];
    else if (k == "--emit-icache" && i + 1 < argc) emit = argv[++i];
    else {
      fprintf(stderr,
              "usage: bwchip --emit-icache DIR\n"
              "       bwchip [--out DIR] [--icache DIR_OF_CUBINS] [--only icache|const|tlb|l1|l2]\n");
      return 2;
    }
  }
  if (!emit.empty()) return chip::emit_icache(emit);
  OK(cudaSetDevice(0));
  cudaDeviceProp p;
  OK(cudaGetDeviceProperties(&p, 0));
  chip::Dev d;
  d.sms = p.multiProcessorCount;
  d.l2 = p.l2CacheSize;
  int clk = 0;
  OK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
  std::string j = "{\"name\":\"" + std::string(p.name) + "\",\"sms\":" + std::to_string(d.sms) +
                  ",\"l2_bytes\":" + std::to_string(p.l2CacheSize) + ",\"regs_per_sm\":" +
                  std::to_string(p.regsPerMultiprocessor) + ",\"smem_per_sm\":" +
                  std::to_string(p.sharedMemPerMultiprocessor) + ",\"max_threads_per_sm\":" +
                  std::to_string(p.maxThreadsPerMultiProcessor) + ",\"max_blocks_per_sm\":" +
                  std::to_string(p.maxBlocksPerMultiProcessor) + ",\"clock_khz_attribute\":" + std::to_string(clk) + ",";
  // A sweep whose gate fails contributes its name to "failed" and nothing else; the rest still run.
  std::string failed;
  auto sweep = [&](const char* name, auto run) {
    if (!only.empty() && only != name) return;
    std::string s;
    if (run(s)) failed += std::string(failed.empty() ? "\"" : ",\"") + name + "\"";
    else j += s;
  };
  if (!icache_dir.empty()) sweep("icache", [&](std::string& s) { return chip::run_icache(s, icache_dir); });
  sweep("const", [&](std::string& s) { return chip::run_const(s); });
  sweep("tlb", [&](std::string& s) { return chip::run_tlb(s); });
  sweep("l1", [&](std::string& s) { return chip::run_l1(s); });
  sweep("l2", [&](std::string& s) { return chip::run_l2(s, d); });
  j += "\"failed\":[" + failed + "],\"passed\":" + (failed.empty() ? "true" : "false") + "}";
  FILE* f = fopen((out + "/chip.json").c_str(), "w");
  fprintf(f, "%s\n", j.c_str());
  fclose(f);
  printf("%s\n", j.c_str());
  return failed.empty() ? 0 : 3;
}
