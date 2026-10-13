// Filler Pearl-C tiles for the decode step's idle tensor cores (lane ra-fill-compute), on sm_120a.
//
// The tile is `pearl-c-sm120-v1`'s 64 x 64 chain and peel, word for word gemm_body<M_PEARLC, 64, 64, 4> of
// benchmarks/pouw/pearl_c_sm120/pearl_c_sm120.cu (included below, so its helpers and its own gemm_pearlc_d are in this
// library): C~ = the E4M3 chain of A'·B~^T with a +0 accumulator per G = 4 atoms promoted by one FP32 add, then U = C~ + P_A·P_B^T
// in four BF16 k16 steps, and y = bf16(U ia ib).  gemm_body walks the whole problem with blockIdx and gridDim, so it cannot run
// a tile range or a work queue; `tile` below is its loop body for one tile, so a filler can be cut anywhere.
//
// Args as gemm_pearlc_d's: p0 A' (M x K), p1 B~ (N x K), p2 C~, p3 U (M x N FP32), p4 y (M x N BF16), p5 P_A (M x 128),
// p6 P_B (N x 128), p7 ia_a (M), p8 ia_b (N); i0 M, i1 N, i2 K, i3 the raster group.  The filler adds p11 a log of each tile's
// end (u64 %globaltimer, one per queue position), p12 the queue counter (u32), p13 the stop flag (u32, pinned host memory
// so the host can end a run whatever the device is doing), p14 the fold slots (F_FOLD); i4 the first queue position, i5 the
// count (a chunk) or the queue's limit (persistent), i6 the persistent throttle's idle ns a tile.  Queue positions past
// M/T * N/T (T the tile) wrap onto the problem again.
#include "../pouw/pearl_c_sm120/pearl_c_sm120.cu"

DEV uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// T the CTA's square tile: 64 (the protocol's, gemm_pearlc_d's) or 128 (four protocol tiles a CTA, each output's chain and peel
// those of gemm_body<M_PEARLC, 128, 128, 4>, which are gemm_pearlc_d's word for word: the per-element arithmetic doesn't
// depend on the tile; the operand bytes a FLOP halve)
constexpr int FG = 4;
template <int T> constexpr int f_smem() { return gemm_smem<M_PEARLC, T, T>(); }

// What a tile writes: F_ALL C~, U and y (gemm_pearlc_d's buffers, the transcript the verifier's tile leaves read); F_Y y alone;
// F_FOLD only the XOR of its C~ words and of its U words into p14[2 t], p14[2 t + 1] (a stand-in for an in-kernel tile leaf,
// so the filler's HBM writes are a leaf's, not the words': not the verifier's bytes).
enum { F_ALL, F_Y, F_FOLD };

DEV uint32_t warp_xor(uint32_t v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v ^= __shfl_xor_sync(~0u, v, o);
  return v;
}

// y over two adjacent n8 fragments of one row, w0 and w1 the y word (two BF16) a lane holds in each, stored so a quad writes
// one whole 32-byte sector: after two quad shuffles lane q holds the row's BF16 4q..4q+3 of the 16.  gemm_body's store_y
// writes each lane's 4 bytes where it computed them, half a sector a quad, so every y sector is written twice.
DEV void store_y_sector(uint16_t* p, uint32_t w0, uint32_t w1, int q) {
  const int base = (threadIdx.x & 31) & ~3, lo = 2 * (q & 1), hi = q >> 1;
  const uint32_t x = __shfl_sync(~0u, (q & 1) ? w1 : w0, base + lo + hi);
  const uint32_t z = __shfl_sync(~0u, (q & 1) ? w0 : w1, base + lo + 1 - hi);
  *reinterpret_cast<uint2*>(p + 4 * q) = hi ? make_uint2(z, x) : make_uint2(x, z);
}
DEV uint32_t y_word(float a, float b, float sr, float sc0, float sc1) {
  return f2bf(__fmul_rn(__fmul_rn(a, sr), sc0)) | (uint32_t)f2bf(__fmul_rn(__fmul_rn(b, sr), sc1)) << 16;
}

// One T x T tile (tm, tn): gemm_body<M_PEARLC, T, T, 4>'s loop body, unchanged but for its indices and what it stores.
template <int STORE, int T> DEV void tile(const Args& a, uint8_t* smem, int tm, int tn, int t) {
  constexpr int TM = T, BN = T, G = FG;
  constexpr int WN = 32, WARPS_N = BN / WN, WARPS_M = 8 / WARPS_N, WM = TM / WARPS_M, MI = WM / 16, NI = WN / 8;
  constexpr int PERIOD = G / 2;
  constexpr uint32_t A_ST = TM * BK, B_ST = BN * BK;
  uint8_t* sA = smem;
  uint8_t* sB = smem + STAGES * A_ST;
  uint8_t* sPA = sB + STAGES * B_ST;
  uint8_t* sPB = sPA + TM * PEEL_ROW;
  const int N = a.i[1], K = a.i[2], KB = K / BK;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
  const uint8_t* A = reinterpret_cast<const uint8_t*>(a.p[0]);
  const uint8_t* B = reinterpret_cast<const uint8_t*>(a.p[1]);
  float* C = reinterpret_cast<float*>(a.p[2]);
  float* U = reinterpret_cast<float*>(a.p[3]);
  uint16_t* Y = reinterpret_cast<uint16_t*>(a.p[4]);
  const float* rs = reinterpret_cast<const float*>(a.p[7]);
  const float* cs = reinterpret_cast<const float*>(a.p[8]);
  const int a_row = wm * WM + (lane & 15), a_half = lane >> 4;
  const int b_row = wn * WN + (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;
  const int g = lane >> 2, q = lane & 3;

  const uint8_t* ga = A + (size_t)tm * TM * K;
  const uint8_t* gb = B + (size_t)tn * BN * K;
  peel_rows<TM>(sPA, reinterpret_cast<const uint8_t*>(a.p[5]) + (size_t)tm * TM * PEEL_ROW, tid);
  peel_rows<BN>(sPB, reinterpret_cast<const uint8_t*>(a.p[6]) + (size_t)tn * BN * PEEL_ROW, tid);
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KB) {
      stage_rows<TM>(sA + s * A_ST, ga + s * BK, K, tid);
      stage_rows<BN>(sB + s * B_ST, gb + s * BK, K, tid);
    }
    cp_commit();
  }
  float acc[MI][NI][4], total[MI][NI][4];
#pragma unroll
  for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int ni = 0; ni < NI; ni++)
#pragma unroll
      for (int e = 0; e < 4; e++) { acc[mi][ni][e] = 0.0f; total[mi][ni][e] = 0.0f; }

  for (int kb = 0; kb < KB; kb++) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nk = kb + STAGES - 1, s = nk % STAGES;
      if (nk < KB) {
        stage_rows<TM>(sA + s * A_ST, ga + (size_t)nk * BK, K, tid);
        stage_rows<BN>(sB + s * B_ST, gb + (size_t)nk * BK, K, tid);
      }
      cp_commit();
    }
    if (kb % PERIOD == 0) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
          for (int e = 0; e < 4; e++) acc[mi][ni][e] = 0.0f;
    }
    const uint32_t a0 = sa(sA + (kb % STAGES) * A_ST), b0 = sa(sB + (kb % STAGES) * B_ST);
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      uint32_t fa[MI][4], fb[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; mi++) ldsm4(fa[mi], a0 + swz64(a_row + 16 * mi, 2 * kk + a_half));
#pragma unroll
      for (int nj = 0; nj < NI / 2; nj++) {
        uint32_t r[4];
        ldsm4(r, b0 + swz64(b_row + 16 * nj, 2 * kk + b_half));
        fb[2 * nj][0] = r[0]; fb[2 * nj][1] = r[1]; fb[2 * nj + 1][0] = r[2]; fb[2 * nj + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) qmma(acc[mi][ni], fa[mi], fb[ni]);
    }
    if (kb % PERIOD == PERIOD - 1) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
          for (int e = 0; e < 4; e++) total[mi][ni][e] = __fadd_rn(total[mi][ni][e], acc[mi][ni][e]);
    }
  }
  cp_wait<0>();
  __syncthreads();

  const int row0 = tm * TM + wm * WM + g, col0 = tn * BN + wn * WN + 2 * q;
  uint32_t fc = 0, fu = 0;
#pragma unroll
  for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
      const int row = row0 + 16 * mi, col = col0 + 8 * ni;
      if constexpr (STORE == F_ALL) {
        store2(C + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1]);
        store2(C + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3]);
      }
      if constexpr (STORE == F_FOLD)
#pragma unroll
        for (int e = 0; e < 4; e++) fc ^= __float_as_uint(total[mi][ni][e]);
    }
  const uint32_t pa0 = sa(sPA), pb0 = sa(sPB);
#pragma unroll
  for (int s = 0; s < 4; s++) {
    uint32_t fa[MI][4], fb[NI][2];
#pragma unroll
    for (int mi = 0; mi < MI; mi++) ldsm4(fa[mi], pa0 + swz128(a_row + 16 * mi, 2 * s + a_half));
#pragma unroll
    for (int nj = 0; nj < NI / 2; nj++) {
      uint32_t r[4];
      ldsm4(r, pb0 + swz128(b_row + 16 * nj, 2 * s + b_half));
      fb[2 * nj][0] = r[0]; fb[2 * nj][1] = r[1]; fb[2 * nj + 1][0] = r[2]; fb[2 * nj + 1][1] = r[3];
    }
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) hmma(total[mi][ni], fa[mi], fb[ni]);
  }
#pragma unroll
  for (int mi = 0; mi < MI; mi++) {
    const int row = row0 + 16 * mi;
    const float r0 = rs[row], r1 = rs[row + 8];
    uint32_t w[NI][2];
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
      const int col = col0 + 8 * ni;
      const float c0 = cs[col], c1 = cs[col + 1];
      if constexpr (STORE == F_ALL) {
        store2(U + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1]);
        store2(U + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3]);
      }
      if constexpr (STORE != F_FOLD) {
        w[ni][0] = y_word(total[mi][ni][0], total[mi][ni][1], r0, c0, c1);
        w[ni][1] = y_word(total[mi][ni][2], total[mi][ni][3], r1, c0, c1);
        if (ni % 2 == 1) {
          const int span = col0 - 2 * q + 8 * (ni - 1);
          store_y_sector(Y + (size_t)row * N + span, w[ni - 1][0], w[ni][0], q);
          store_y_sector(Y + (size_t)(row + 8) * N + span, w[ni - 1][1], w[ni][1], q);
        }
      } else {
        (void)r0; (void)r1; (void)c0; (void)c1;
#pragma unroll
        for (int e = 0; e < 4; e++) fu ^= __float_as_uint(total[mi][ni][e]);
      }
    }
  }
  if constexpr (STORE == F_FOLD) {
    fc = warp_xor(fc);
    fu = warp_xor(fu);
    if (lane == 0) {
      atomicXor(reinterpret_cast<unsigned*>(a.p[14]) + 2 * t, fc);
      atomicXor(reinterpret_cast<unsigned*>(a.p[14]) + 2 * t + 1, fu);
    }
  }
  __syncthreads();
}

// Queue position t: tile t mod (M/T * N/T) of the problem; the log's slot t gets its end time.  The fold slots are per tile
// of the problem, so a fold over a queue that wraps XORs each tile in once per pass.
template <int STORE, int T> DEV void tile_at(const Args& a, uint8_t* smem, int t) {
  const int tiles_m = a.i[0] / T, tiles_n = a.i[1] / T, tiles = tiles_m * tiles_n;
  int tm, tn;
  raster(t % tiles, tiles_m, tiles_n, a.i[3], tm, tn);
  tile<STORE, T>(a, smem, tm, tn, t % tiles);
  if (threadIdx.x == 0 && a.p[11]) reinterpret_cast<uint64_t*>(a.p[11])[t] = gtimer();
}

DEV bool stopped(const Args& a) { return a.p[13] && *reinterpret_cast<volatile unsigned*>(a.p[13]); }

// A chunk: queue positions i4 .. i4 + i5 - 1, strided over the grid; a CTA leaves at the stop flag (checked per tile).
template <int STORE, int T> DEV void chunk_body(const Args& a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int halt;
  for (int t = blockIdx.x; t < a.i[5]; t += gridDim.x) {
    if (threadIdx.x == 0) halt = stopped(a);
    __syncthreads();
    if (halt) return;
    tile_at<STORE, T>(a, smem, a.i[4] + t);
  }
}

// Persistent: each CTA takes the queue's next position until the stop flag is set or the queue reaches i5; i6 > 0 a throttle,
// each CTA idle for i6 ns after each tile (asleep: no instruction issued, no memory traffic).
template <int STORE, int T> DEV void persist_body(const Args& a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int next;
  unsigned* counter = reinterpret_cast<unsigned*>(a.p[12]);
  for (;;) {
    if (threadIdx.x == 0) next = stopped(a) ? -1 : (int)atomicAdd(counter, 1u);
    __syncthreads();
    const int t = next;
    __syncthreads();
    if (t < 0 || t >= a.i[5]) return;
    tile_at<STORE, T>(a, smem, t);
    if (a.i[6] > 0) {
      const uint64_t t1 = gtimer();
      while (gtimer() - t1 < (uint64_t)a.i[6]) __nanosleep(2000);
    }
  }
}

// fill_<chunk|persist>_<MINB>_<STORE>: the 64 x 64 tile, MINB the launch bound's CTAs per SM (2 caps the registers at 128);
// fill_<chunk|persist>128_1_<STORE>: the 128 x 128 tile (96 KB of shared memory: one CTA an SM)
#define FILL(MINB, STORE)                                                                                          \
  extern "C" __global__ void __launch_bounds__(NT, MINB) fill_chunk_##MINB##_##STORE(const Args a) {                \
    chunk_body<STORE, 64>(a);                                                                                     \
  }                                                                                                               \
  extern "C" __global__ void __launch_bounds__(NT, MINB) fill_persist_##MINB##_##STORE(const Args a) {              \
    persist_body<STORE, 64>(a);                                                                                   \
  }
#define FILL128(STORE)                                                                                             \
  extern "C" __global__ void __launch_bounds__(NT, 1) fill_chunk128_1_##STORE(const Args a) {                       \
    chunk_body<STORE, 128>(a);                                                                                    \
  }                                                                                                               \
  extern "C" __global__ void __launch_bounds__(NT, 1) fill_persist128_1_##STORE(const Args a) {                     \
    persist_body<STORE, 128>(a);                                                                                  \
  }
FILL(1, 0)
FILL(1, 1)
FILL(1, 2)
FILL(2, 0)
FILL(2, 1)
FILL(2, 2)
FILL128(0)
FILL128(1)
FILL128(2)

// p0 a u64 slot: %globaltimer when the kernel runs
extern "C" __global__ void stamp(const Args a) { *reinterpret_cast<uint64_t*>(a.p[0]) = gtimer(); }
// p0 a u32 flag, set to i0
extern "C" __global__ void set_flag(const Args a) { *reinterpret_cast<volatile unsigned*>(a.p[0]) = (unsigned)a.i[0]; }
// p0 a u32 per CTA: the SM it ran on, after holding it for i0 ns (so a partition's SMs all take CTAs)
extern "C" __global__ void smid_of(const Args a) {
  const uint64_t t0 = gtimer();
  while (gtimer() - t0 < (uint64_t)a.i[0]) {
  }
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  if (threadIdx.x == 0) reinterpret_cast<unsigned*>(a.p[0])[blockIdx.x] = s;
}

// ---------------------------------------------------------------- host side (ctypes)
#include <cuda_runtime.h>
#include <cuda.h>

static int launch(const void* fn, int grid, int block, int smem, const Args* a, cudaStream_t s) {
  void* argv[] = {const_cast<Args*>(a)};
  // static shared memory counts against the default 48 KB too, so opt in whenever there is dynamic shared memory
  if (smem > 0 && cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return -1;
  return (int)cudaLaunchKernel(fn, dim3(grid), dim3(block), argv, (size_t)smem, s);
}

static const void* FILLS[2][2][3] = {
    {{(const void*)fill_chunk_1_0, (const void*)fill_chunk_1_1, (const void*)fill_chunk_1_2},
     {(const void*)fill_chunk_2_0, (const void*)fill_chunk_2_1, (const void*)fill_chunk_2_2}},
    {{(const void*)fill_persist_1_0, (const void*)fill_persist_1_1, (const void*)fill_persist_1_2},
     {(const void*)fill_persist_2_0, (const void*)fill_persist_2_1, (const void*)fill_persist_2_2}}};
static const void* FILLS128[2][3] = {
    {(const void*)fill_chunk128_1_0, (const void*)fill_chunk128_1_1, (const void*)fill_chunk128_1_2},
    {(const void*)fill_persist128_1_0, (const void*)fill_persist128_1_1, (const void*)fill_persist128_1_2}};
// minb 1 or 2: the 64 x 64 tile at that launch bound; 128: the 128 x 128 tile
static const void* fill_fn(int persist, int minb, int store) {
  if (store < 0 || store > 2) return nullptr;
  if (minb == 128) return FILLS128[persist ? 1 : 0][store];
  if (minb < 1 || minb > 2) return nullptr;
  return FILLS[persist ? 1 : 0][minb - 1][store];
}
static int smem_of(int minb) { return minb == 128 ? f_smem<128>() : f_smem<64>(); }

extern "C" int fill_smem(int minb) { return smem_of(minb); }
// persist 0: a chunk; 1: persistent CTAs.  Returns the cudaError_t, or -1 for an unknown kernel.
extern "C" int fill_launch(int persist, int minb, int store, int grid, const Args* a, void* stream) {
  const void* fn = fill_fn(persist, minb, store);
  return fn ? launch(fn, grid, NT, smem_of(minb), a, (cudaStream_t)stream) : -1;
}
// PoUW's own 64 x 64 kernel on the whole problem (the reference arm at the timing shape)
extern "C" int pearlc_d(int grid, const Args* a, void* stream) {
  return launch((const void*)gemm_pearlc_d, grid, NT, gemm_smem<M_PEARLC, 64, 64>(), a, (cudaStream_t)stream);
}
extern "C" int stamp_at(void* slot, void* stream) {
  Args a{};
  a.p[0] = (uint64_t)slot;
  return launch((const void*)stamp, 1, 1, 0, &a, (cudaStream_t)stream);
}
extern "C" int smid_at(void* out, int grid, int hold_ns, void* stream) {
  Args a{};
  a.p[0] = (uint64_t)out;
  a.i[0] = hold_ns;
  return launch((const void*)smid_of, grid, 32, 0, &a, (cudaStream_t)stream);
}
extern "C" int flag_set(void* flag, int v, void* stream) {
  Args a{};
  a.p[0] = (uint64_t)flag;
  a.i[0] = v;
  return launch((const void*)set_flag, 1, 1, 0, &a, (cudaStream_t)stream);
}
// Green contexts: the device's SMs split into the step's (at least n_step, as the driver rounds) and the rest, a stream in
// each (the driver API through dlopen, so the library needs no libcuda stub at link time).  got: the step's SMs, the
// filler's, the minimum partition and the co-scheduled alignment.  *s_fill is null when nothing remains.  Returns 0, or
// 1000 * the failing stage + its CUresult (stage 1 a missing driver symbol).
#include <dlfcn.h>
template <class F> static F drv(const char* name) {
  static void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  return h ? reinterpret_cast<F>(dlsym(h, name)) : nullptr;
}
static int green(CUdevResource* r, CUdevice dev, int prio, CUstream* s) {
  auto gen = drv<decltype(&cuDevResourceGenerateDesc)>("cuDevResourceGenerateDesc");
  auto mk = drv<decltype(&cuGreenCtxCreate)>("cuGreenCtxCreate");
  auto st = drv<decltype(&cuGreenCtxStreamCreate)>("cuGreenCtxStreamCreate");
  if (!gen || !mk || !st) return 1000;
  CUdevResourceDesc d;
  CUgreenCtx g;
  int e;
  if ((e = gen(&d, r, 1))) return 4000 + e;
  if ((e = mk(&g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM))) return 5000 + e;
  if ((e = st(s, g, CU_STREAM_NON_BLOCKING, prio))) return 6000 + e;
  return 0;
}
extern "C" int gc_split(int n_step, int prio_step, int prio_fill, void** s_step, void** s_fill, int* got) {
  auto dget = drv<decltype(&cuDeviceGet)>("cuDeviceGet");
  auto res = drv<decltype(&cuDeviceGetDevResource)>("cuDeviceGetDevResource");
  auto split = drv<decltype(&cuDevSmResourceSplitByCount)>("cuDevSmResourceSplitByCount");
  if (!dget || !res || !split) return 1000;
  cudaFree(nullptr);
  CUdevice dev;
  CUdevResource all{}, part{}, rest{};
  unsigned nb = 1;
  int e;
  if ((e = dget(&dev, 0))) return 2000 + e;
  if ((e = res(dev, &all, CU_DEV_RESOURCE_TYPE_SM))) return 2000 + e;
  if ((e = split(&part, &nb, &all, &rest, 0, (unsigned)n_step))) return 3000 + e;
  got[0] = (int)part.sm.smCount;
  got[1] = (int)rest.sm.smCount;
  got[2] = (int)all.sm.minSmPartitionSize;
  got[3] = (int)all.sm.smCoscheduledAlignment;
  CUstream a = nullptr, b = nullptr;
  if ((e = green(&part, dev, prio_step, &a))) return e;
  if (rest.sm.smCount && (e = green(&rest, dev, prio_fill, &b))) return e;
  *s_step = a;
  *s_fill = b;
  return 0;
}

// The pump: a native thread (no GIL, beside an engine that runs Python) that keeps `depth` chunks of `per` tiles a CTA over
// `grid` CTAs queued on `stream`, each the next queue positions, so the filler never holds the device for longer than
// `depth` chunks: a device-wide synchronize (cudaFree, torch.cuda.synchronize) beside a persistent filler would never
// return.  The stop flag is not set: fill_pump_stop stops launching and waits for the chunks in flight, so every position
// launched is a tile computed, and returns their count.
#include <atomic>
#include <thread>
#include <vector>
static std::thread pump_thread;
static std::atomic<bool> pump_on{false};
static std::atomic<long long> pump_done_tiles{0};
static int pump_rc = 0;
extern "C" int fill_pump_start(int minb, int store, int grid, int per, int depth, const Args* a, void* stream) {
  const void* fn = fill_fn(0, minb, store);
  if (!fn || pump_on.load() || depth < 1) return -1;
  Args args = *a;
  cudaStream_t s = (cudaStream_t)stream;
  pump_done_tiles = 0;
  pump_rc = 0;
  pump_on = true;
  pump_thread = std::thread([=]() mutable {
    std::vector<cudaEvent_t> ev(depth);
    std::vector<long long> n(depth, 0);
    for (auto& e : ev) cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync);
    const long long chunk = (long long)grid * per;
    long long pos = args.i[4];
    for (long long k = 0; pump_on.load(); k++) {
      const int slot = (int)(k % depth);
      if (k >= depth) {
        cudaEventSynchronize(ev[slot]);
        pump_done_tiles += n[slot];
        n[slot] = 0;
      }
      args.i[4] = (int)pos;
      args.i[5] = (int)chunk;
      if ((pump_rc = launch(fn, grid, NT, smem_of(minb), &args, s))) break;
      cudaEventRecord(ev[slot], s);
      n[slot] = chunk;
      pos += chunk;
    }
    for (int j = 0; j < depth; j++) cudaEventSynchronize(ev[j]);
    long long rest = 0;
    for (int j = 0; j < depth; j++) rest += n[j];
    pump_done_tiles += rest;
    for (auto& e : ev) cudaEventDestroy(e);
  });
  return 0;
}
extern "C" long long fill_pump_tiles() { return pump_done_tiles.load(); }
extern "C" long long fill_pump_stop(int* rc) {
  pump_on = false;
  if (pump_thread.joinable()) pump_thread.join();
  if (rc) *rc = pump_rc;
  return pump_done_tiles.load();
}

// the filler's resident CTAs per SM at its dynamic shared memory, and its registers per thread
extern "C" int fill_occupancy(int persist, int minb, int store, int* regs) {
  int n = 0;
  const void* fn = fill_fn(persist, minb, store);
  if (!fn) return -1;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_of(minb));
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, NT, smem_of(minb));
  cudaFuncAttributes at;
  if (regs && cudaFuncGetAttributes(&at, fn) == cudaSuccess) *regs = at.numRegs;
  return n;
}
