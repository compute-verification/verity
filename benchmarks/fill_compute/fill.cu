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
// count (a chunk) or the queue's limit (persistent).  Queue positions past M/64 * N/64 wrap onto the problem again.
#include "../pouw/pearl_c_sm120/pearl_c_sm120.cu"

DEV uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

constexpr int FT = 64, FB = 64, FG = 4, F_SMEM = gemm_smem<M_PEARLC, FT, FB>();

// What a tile writes: F_ALL C~, U and y (gemm_pearlc_d's buffers, the transcript the verifier's tile leaves read); F_Y y alone;
// F_FOLD only the XOR of its C~ words and of its U words into p14[2 t], p14[2 t + 1] (a stand-in for an in-kernel tile leaf,
// so the filler's HBM writes are a leaf's, not the words': not the verifier's bytes).
enum { F_ALL, F_Y, F_FOLD };

DEV uint32_t warp_xor(uint32_t v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v ^= __shfl_xor_sync(~0u, v, o);
  return v;
}

// One 64 x 64 tile (tm, tn): gemm_body<M_PEARLC, 64, 64, 4>'s loop body, unchanged but for its indices and what it stores.
template <int STORE> DEV void tile(const Args& a, uint8_t* smem, int tm, int tn, int t) {
  constexpr int TM = FT, BN = FB, G = FG;
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
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
      const int col = col0 + 8 * ni;
      const float c0 = cs[col], c1 = cs[col + 1];
      if constexpr (STORE == F_ALL) {
        store2(U + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1]);
        store2(U + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3]);
      }
      if constexpr (STORE != F_FOLD) {
        store_y(Y + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1], r0, c0, c1);
        store_y(Y + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3], r1, c0, c1);
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

// Queue position t: tile t mod (M/64 * N/64) of the problem; the log's slot t gets its end time.  The fold slots are per tile
// of the problem, so a fold over a queue that wraps XORs each tile in once per pass.
template <int STORE> DEV void tile_at(const Args& a, uint8_t* smem, int t) {
  const int tiles_m = a.i[0] / FT, tiles_n = a.i[1] / FB, tiles = tiles_m * tiles_n;
  int tm, tn;
  raster(t % tiles, tiles_m, tiles_n, a.i[3], tm, tn);
  tile<STORE>(a, smem, tm, tn, t % tiles);
  if (threadIdx.x == 0 && a.p[11]) reinterpret_cast<uint64_t*>(a.p[11])[t] = gtimer();
}

DEV bool stopped(const Args& a) { return a.p[13] && *reinterpret_cast<volatile unsigned*>(a.p[13]); }

// A chunk: queue positions i4 .. i4 + i5 - 1, strided over the grid; a CTA leaves at the stop flag (checked per tile).
template <int STORE> DEV void chunk_body(const Args& a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int halt;
  for (int t = blockIdx.x; t < a.i[5]; t += gridDim.x) {
    if (threadIdx.x == 0) halt = stopped(a);
    __syncthreads();
    if (halt) return;
    tile_at<STORE>(a, smem, a.i[4] + t);
  }
}

// Persistent: each CTA takes the queue's next position until the stop flag is set or the queue reaches i5.
template <int STORE> DEV void persist_body(const Args& a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int next;
  unsigned* counter = reinterpret_cast<unsigned*>(a.p[12]);
  for (;;) {
    if (threadIdx.x == 0) next = stopped(a) ? -1 : (int)atomicAdd(counter, 1u);
    __syncthreads();
    const int t = next;
    __syncthreads();
    if (t < 0 || t >= a.i[5]) return;
    tile_at<STORE>(a, smem, t);
  }
}

// fill_<chunk|persist>_<MINB>_<STORE>: MINB the launch bound's CTAs per SM (2 caps the registers at 128)
#define FILL(MINB, STORE)                                                                                          \
  extern "C" __global__ void __launch_bounds__(NT, MINB) fill_chunk_##MINB##_##STORE(const Args a) {                \
    chunk_body<STORE>(a);                                                                                         \
  }                                                                                                               \
  extern "C" __global__ void __launch_bounds__(NT, MINB) fill_persist_##MINB##_##STORE(const Args a) {              \
    persist_body<STORE>(a);                                                                                       \
  }
FILL(1, 0)
FILL(1, 1)
FILL(1, 2)
FILL(2, 0)
FILL(2, 1)
FILL(2, 2)

// p0 a u64 slot: %globaltimer when the kernel runs
extern "C" __global__ void stamp(const Args a) { *reinterpret_cast<uint64_t*>(a.p[0]) = gtimer(); }
// p0 a u32 flag, set to i0
extern "C" __global__ void set_flag(const Args a) { *reinterpret_cast<volatile unsigned*>(a.p[0]) = (unsigned)a.i[0]; }

// ---------------------------------------------------------------- host side (ctypes)
#include <cuda_runtime.h>

static int launch(const void* fn, int grid, int block, int smem, const Args* a, cudaStream_t s) {
  void* argv[] = {const_cast<Args*>(a)};
  if (smem > 48 * 1024 && cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return -1;
  return (int)cudaLaunchKernel(fn, dim3(grid), dim3(block), argv, (size_t)smem, s);
}

static const void* FILLS[2][2][3] = {
    {{(const void*)fill_chunk_1_0, (const void*)fill_chunk_1_1, (const void*)fill_chunk_1_2},
     {(const void*)fill_chunk_2_0, (const void*)fill_chunk_2_1, (const void*)fill_chunk_2_2}},
    {{(const void*)fill_persist_1_0, (const void*)fill_persist_1_1, (const void*)fill_persist_1_2},
     {(const void*)fill_persist_2_0, (const void*)fill_persist_2_1, (const void*)fill_persist_2_2}}};
static const void* fill_fn(int persist, int minb, int store) {
  if (minb < 1 || minb > 2 || store < 0 || store > 2) return nullptr;
  return FILLS[persist ? 1 : 0][minb - 1][store];
}

extern "C" int fill_smem() { return F_SMEM; }
// persist 0: a chunk; 1: persistent CTAs.  Returns the cudaError_t, or -1 for an unknown kernel.
extern "C" int fill_launch(int persist, int minb, int store, int grid, const Args* a, void* stream) {
  const void* fn = fill_fn(persist, minb, store);
  return fn ? launch(fn, grid, NT, F_SMEM, a, (cudaStream_t)stream) : -1;
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
extern "C" int flag_set(void* flag, int v, void* stream) {
  Args a{};
  a.p[0] = (uint64_t)flag;
  a.i[0] = v;
  return launch((const void*)set_flag, 1, 1, 0, &a, (cudaStream_t)stream);
}
// the filler's resident CTAs per SM at its dynamic shared memory, and its registers per thread
extern "C" int fill_occupancy(int persist, int minb, int store, int* regs) {
  int n = 0;
  const void* fn = fill_fn(persist, minb, store);
  if (!fn) return -1;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, F_SMEM);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, NT, F_SMEM);
  cudaFuncAttributes at;
  if (regs && cudaFuncGetAttributes(&at, fn) == cudaSuccess) *regs = at.numRegs;
  return n;
}
