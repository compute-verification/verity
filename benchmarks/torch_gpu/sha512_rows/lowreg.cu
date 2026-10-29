// SHA-512's rate against the registers a thread may hold, for a hash warpgroup beside MMA warpgroups (lowreg.cuh). Each
// thread hashes sha512/row/v2 rows (rows.cuh's row_digest: P(8 nbytes)'s midstate, then the row's chunks, then the padding
// chunk's rounds alone, compress_kw) from a 128-row tile in shared memory, 512 bytes a row (256 bf16 columns: a 128 x 256 O
// tile), its stride 528 bytes so a warp's 16-byte loads of its rows' words take the fewest wavefronts. A row is TILES
// passes over the tile row, so a thread chains 4 TILES + 1 compressions, as a hash warpgroup that owns a row of the CTA's
// panel chains it through the panel's column tiles.
//
// Variants (spec, NAMES): where the schedule W (32 registers) and the chaining value (16) live, and one chain or two a thread:
//   r, r0, r2   sx::compress<0, 1 | 0 | 2>: both in registers (r is sha512x.cuh's default)
//   rs          W in registers, the chaining value in shared memory (lowreg.cuh's rounds)
//   s, ss       W in shared memory ([word][thread]); the chaining value in registers, or in shared memory too
//   d, ds, dr   two rows a thread interleaved (rounds2): W in registers or shared memory, the chaining values in shared
//               memory (d, ds) or registers (dr)
//   rs<SSss>    rs with sha512x.cuh's V: the pipe of S0, S1, s0 and s1's shifts (README "Compression variants"), some on
//               IMAD, which issues to the FMA pipe beside the ALU: with one warp a scheduler the ALU's throughput bounds
//               rs, and the FMA pipe is idle
// The rest use V 0.
// Each at the register caps CAPS (__maxnreg__; 255 none), launched as one CTA an SM (its shared memory) of 1, 2 or 4
// warpgroups.
//
//   lowreg check TILES     one launch of every kernel at 1 and 2 warpgroups: JSON lines with CTA 0's digests (thread
//                          order, a thread's chains in turn), for lowreg_check.py to compare with hashlib
//   lowreg rates LAUNCHES TILES [WGS...]
//                          one JSON line a kernel and warpgroup count: registers, local memory (spills), compressions a
//                          second, compressions an SM-clock, the SM clock (NVML after each launch), medians of LAUNCHES
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "lowreg.cuh"

using u64 = uint64_t;
using u32 = uint32_t;

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

constexpr int ROWB = 512, TSTRIDE = 528, TROWS = 128, TILE_BYTES = TROWS * TSTRIDE, SEED = 41;
constexpr int MIN_SMEM = 120 * 1024;  // more than half an SM's shared memory: one CTA an SM

struct VarSpec {
  int chains, st_smem, w_smem, roll, sx, v;
};
#define LR_V(a, b, c, d) ((a) | (b) << 2 | (c) << 4 | (d) << 6)
static const char* const NAMES[] = {"r", "r0", "r2", "rs", "s", "ss", "d", "ds", "dr", "rs0011", "rs0033", "rs1100", "rs3300"};
__host__ __device__ constexpr VarSpec spec(int var) {
  switch (var) {
    case 0: return {1, 0, 0, 1, 1, 0};
    case 1: return {1, 0, 0, 0, 1, 0};
    case 2: return {1, 0, 0, 2, 1, 0};
    case 3: return {1, 1, 0, 1, 0, 0};
    case 4: return {1, 0, 1, 1, 0, 0};
    case 5: return {1, 1, 1, 1, 0, 0};
    case 6: return {2, 1, 0, 1, 0, 0};
    case 7: return {2, 1, 1, 1, 0, 0};
    case 8: return {2, 0, 0, 1, 0, 0};
    case 9: return {1, 1, 0, 1, 0, LR_V(0, 0, 1, 1)};
    case 10: return {1, 1, 0, 1, 0, LR_V(0, 0, 3, 3)};
    case 11: return {1, 1, 0, 1, 0, LR_V(1, 1, 0, 0)};
    default: return {1, 1, 0, 1, 0, LR_V(3, 3, 0, 0)};
  }
}
#undef LR_V

// the shared memory a CTA of `wgs` warpgroups takes: the tile, then each warpgroup's schedules, then its chaining values
__host__ __device__ constexpr int w_words(const VarSpec& v) { return v.w_smem ? v.chains * 16 * 128 : 0; }
__host__ __device__ constexpr int s_words(const VarSpec& v) { return v.st_smem ? v.chains * 8 * 128 : 0; }
static int smem_of(const VarSpec& v, int wgs) {
  return std::max(MIN_SMEM, TILE_BYTES + wgs * 8 * (w_words(v) + s_words(v)));
}

// one chunk (c) of each of this thread's rows into its chaining values: st (registers) or Ss[(chain 8 + i) 128] (shared)
template <int VAR>
__device__ __forceinline__ void hash_chunk(const uint8_t* tile, const int* rows, int c, u64 (&st)[2][8], u64* Ws, u64* Ss) {
  constexpr VarSpec V = spec(VAR);
  constexpr int C = V.chains;
  if constexpr (V.sx) {
    u64 w[16];
    sx::load_chunk(tile + rows[0] * TSTRIDE + 128 * c, w);
    sx::compress<V.v, V.roll>(st[0], w);
  } else {
    u64 v[C][8];
#pragma unroll
    for (int ch = 0; ch < C; ++ch)
#pragma unroll
      for (int i = 0; i < 8; ++i) v[ch][i] = V.st_smem ? Ss[(ch * 8 + i) * 128] : st[ch][i];
    // the feed-forward reads the chaining value from shared memory again rather than holding it through the rounds
    if constexpr (V.st_smem) asm volatile("" ::: "memory");
    if constexpr (V.w_smem) {
      const sx::Strided<128> W0{Ws}, W1{Ws + 16 * 128};
      sx::load_chunk(tile + rows[0] * TSTRIDE + 128 * c, W0);
      if constexpr (C == 2) {
        sx::load_chunk(tile + rows[1] * TSTRIDE + 128 * c, W1);
        sx::rounds2<0>(v[0], W0, v[1], W1);
      } else {
        sx::rounds<V.v, 1>(v[0], W0);
      }
    } else {
      u64 w0[16], w1[16];
      sx::load_chunk(tile + rows[0] * TSTRIDE + 128 * c, w0);
      if constexpr (C == 2) {
        sx::load_chunk(tile + rows[1] * TSTRIDE + 128 * c, w1);
        sx::rounds2<0>(v[0], w0, v[1], w1);
      } else {
        sx::rounds<V.v, 1>(v[0], w0);
      }
    }
#pragma unroll
    for (int ch = 0; ch < C; ++ch)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if constexpr (V.st_smem)
          Ss[(ch * 8 + i) * 128] += v[ch][i];
        else
          st[ch][i] += v[ch][i];
      }
  }
}

template <int VAR, int CAP>
__global__ void __maxnreg__(CAP) rate(const uint4* __restrict__ gtile, const u64* __restrict__ mid, const u64* __restrict__ kw,
                                      int tiles, u64* __restrict__ out) {
  extern __shared__ __align__(16) uint8_t sm[];
  constexpr VarSpec V = spec(VAR);
  constexpr int C = V.chains;
  const int tid = threadIdx.x, nt = blockDim.x, wg = tid / 128, lane = tid % 128;
  for (int i = tid; i < TROWS * ROWB / 16; i += nt)
    *reinterpret_cast<uint4*>(sm + (i / (ROWB / 16)) * TSTRIDE + 16 * (i % (ROWB / 16))) = gtile[i];
  u64* base = reinterpret_cast<u64*>(sm + TILE_BYTES);
  u64* Ws = base + wg * w_words(V) + lane;
  u64* Ss = base + (nt / 128) * w_words(V) + wg * s_words(V) + lane;
  const int rows[2] = {lane, (lane + 64) % 128};
  u64 st[2][8];
#pragma unroll
  for (int ch = 0; ch < C; ++ch)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (V.st_smem)
        Ss[(ch * 8 + i) * 128] = mid[i];
      else
        st[ch][i] = mid[i];
    }
  __syncthreads();
#pragma unroll 1
  for (int k = 0; k < tiles; ++k)
#pragma unroll 1
    for (int c = 0; c < 4; ++c) hash_chunk<VAR>(sm, rows, c, st, Ws, Ss);
  u64* o = out + ((size_t)blockIdx.x * nt + tid) * 8 * C;
#pragma unroll 1
  for (int ch = 0; ch < C; ++ch) {
    u64 s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = V.st_smem ? Ss[(ch * 8 + i) * 128] : st[ch][i];
    sx::compress_kw<0, 1>(s, kw);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[ch * 8 + i] = s[i];
  }
}

typedef void (*RateFn)(const uint4*, const u64*, const u64*, int, u64*);
struct Kern {
  int var, cap;
  RateFn f;
};
#define CAPS_OF(V) {V, 64, rate<V, 64>}, {V, 72, rate<V, 72>}, {V, 80, rate<V, 80>}, {V, 96, rate<V, 96>}, \
                   {V, 128, rate<V, 128>}, {V, 255, rate<V, 255>}
static const Kern KERNS[] = {CAPS_OF(0), CAPS_OF(1), CAPS_OF(2), CAPS_OF(3),  CAPS_OF(4),  CAPS_OF(5), CAPS_OF(6),
                             CAPS_OF(7), CAPS_OF(8), CAPS_OF(9), CAPS_OF(10), CAPS_OF(11), CAPS_OF(12)};

static u64 splitmix(u64 i) {
  u64 z = (u64)SEED + (i + 1) * 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// P(8 nbytes)'s midstate and the padding chunk's K + W (rows.cuh's row_setup) for rows of nbytes, nbytes % 128 == 0
static void row_setup(u64 nbytes, u64* mid, u64* kw) {
  uint8_t p[128] = {};
  const char* tag = "verity/sha512-row/v2";
  memcpy(p, tag, strlen(tag) + 1);
  for (int i = 0; i < 8; ++i) p[strlen(tag) + 1 + i] = (uint8_t)((8 * nbytes) >> (56 - 8 * i));
  u64 w[16];
  for (int j = 0; j < 16; ++j) {
    w[j] = 0;
    for (int b = 0; b < 8; ++b) w[j] = w[j] << 8 | p[8 * j + b];
  }
  sx::iv(mid);
  sx::compress<0, 1>(mid, w);
  u64 pad[16] = {};
  pad[0] = 0x8000000000000000ull;
  pad[15] = 8 * (128 + nbytes);
  sx::kw_of(pad, kw);
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  double sm() {
    unsigned c = 0;
    if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c);
    return c;
  }
};

static const char* USAGE = "usage: lowreg check TILES | rates LAUNCHES TILES [WGS...]\n";

int main(int argc, char** argv) {
  if (argc < 3) { fputs(USAGE, stderr); return 2; }
  const bool check = !strcmp(argv[1], "check"), rates = !strcmp(argv[1], "rates");
  if (!check && !(rates && argc >= 4)) { fputs(USAGE, stderr); return 2; }
  const int launches = rates ? atoi(argv[2]) : 1, tiles = atoi(argv[rates ? 3 : 2]);
  std::vector<int> wgs_list;
  for (int a = 4; rates && a < argc; ++a) wgs_list.push_back(atoi(argv[a]));
  if (wgs_list.empty()) wgs_list = check ? std::vector<int>{1, 2} : std::vector<int>{1, 2, 4};
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  const int sms = prop.multiProcessorCount;
  std::vector<uint8_t> tile(TROWS * ROWB);
  for (size_t i = 0; i < tile.size() / 8; ++i)
    for (int b = 0; b < 8; ++b) tile[8 * i + b] = (uint8_t)(splitmix(i) >> (8 * b));
  u64 mid[8], kw[80];
  row_setup((u64)tiles * ROWB, mid, kw);
  uint4* gtile;
  u64 *dmid, *dkw, *out;
  CK(cudaMalloc(&gtile, tile.size()));
  CK(cudaMalloc(&dmid, sizeof mid));
  CK(cudaMalloc(&dkw, sizeof kw));
  CK(cudaMalloc(&out, (size_t)sms * 512 * 16 * 8));
  CK(cudaMemcpy(gtile, tile.data(), tile.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dmid, mid, sizeof mid, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dkw, kw, sizeof kw, cudaMemcpyHostToDevice));
  printf("{\"sms\": %d, \"device\": \"%s\", \"seed\": %d, \"tiles\": %d, \"row_bytes\": %d}\n", sms, prop.name, SEED, tiles, ROWB);
  Nv nv;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<u64> host((size_t)512 * 16);
  for (const Kern& k : KERNS) {
    const VarSpec v = spec(k.var);
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, k.f));
    for (int wgs : wgs_list) {
      const int nt = 128 * wgs, smem = smem_of(v, wgs);
      if (smem > (int)prop.sharedMemPerBlockOptin || fa.numRegs * nt > prop.regsPerBlock) continue;
      CK(cudaFuncSetAttribute(k.f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      int per_sm = 0;
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k.f, nt, smem));
      if (per_sm != 1) continue;
      char head[256];
      snprintf(head, sizeof head,
               "\"var\": \"%s\", \"cap\": %d, \"wgs\": %d, \"threads\": %d, \"regs\": %d, \"local_bytes\": %zu, \"chains\": %d",
               NAMES[k.var], k.cap, wgs, nt, fa.numRegs, fa.localSizeBytes, v.chains);
      if (check) {
        CK(cudaMemset(out, 0, (size_t)nt * 16 * 8));
        k.f<<<1, nt, smem>>>(gtile, dmid, dkw, tiles, out);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(host.data(), out, (size_t)nt * v.chains * 64, cudaMemcpyDeviceToHost));
        printf("{%s, \"digests\": \"", head);
        for (int i = 0; i < nt * v.chains * 8; ++i) printf("%016llx", (unsigned long long)host[i]);
        printf("\"}\n");
        fflush(stdout);
        continue;
      }
      std::vector<double> ms, mhz;
      for (int i = 0; i <= launches; ++i) {
        CK(cudaEventRecord(e0));
        k.f<<<sms, nt, smem>>>(gtile, dmid, dkw, tiles, out);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        float m;
        CK(cudaEventElapsedTime(&m, e0, e1));
        if (i) { ms.push_back(m); mhz.push_back(nv.sm()); }
      }
      const double comps = (double)sms * nt * v.chains * (4.0 * tiles + 1), tm = median(ms), f = median(mhz);
      printf("{%s, \"compressions\": %.0f, \"ms\": %.5f, \"gcomp_per_s\": %.4f, \"comp_per_sm_clk\": %.6f, \"mhz\": %.0f}\n", head,
             comps, tm, comps / tm * 1e-6, f > 0 ? comps / (tm * 1e-3 * f * 1e6 * sms) : 0.0, f);
      fflush(stdout);
    }
  }
  return 0;
}
