// The integer tensor-core step on Hopper and datacenter Blackwell, one instruction per tile in eight forms: s8 or u8
// operands, an s32 accumulator, saturating or not. Every output word is the instruction's own function of (C[i][j],
// A[i][:], B[j][:]): no software reduction in between, and operands and words move as raw bits.
//
//   sm_90a   mma_*   mma.sync.aligned.m16n8k32.row.col{.satfinite}.s32.{s8,u8}.{s8,u8}.s32 (one warp a 16 x 8 tile)
//            wg_*    wgmma.mma_async.sync.aligned.m64n8k32.s32.{s8,u8}.{s8,u8}{.satfinite}, A and B from shared memory
//                    (one warpgroup a 64 x 8 tile; the accumulator registers start at C, scale-d 1)
//            wgN_*   the same wgmma at m64nNk32 for N = 64, 128, 256, the N an s8 GEMM's 64 x N warpgroup tile issues
//                    (one warpgroup a 64 x N tile, wgmma.cuh's wrappers)
//   sm_100a  mma_*   the same mma.sync
//            tc_*    tcgen05.mma.cta_group::1.kind::i8, M = 128, N = 32, K = 32, D in tensor memory preloaded with C by
//                    tcgen05.st, enable-input-d 1; the instruction descriptor's saturate bit (bit 3) is the .satfinite
//                    form (one CTA of 4 warps a 128 x 32 tile)
//
//   s8_mma FAMILY CASES OUTDIR    runs FAMILY's eight forms over CASES (s8_mma.py gen) and writes OUTDIR/d_<form>.bin
//
// CASES: "S8MMAPR1", u32 ntiles, u16 M, u16 N, then A (ntiles M 32 bytes), B (ntiles N 32 bytes, k contiguous: the ".col"
// operand, B[j][k]) and C (ntiles M N u32 words, row-major), little-endian. D is C's shape. The shared-memory operands of
// wgmma and tcgen05 are dc.cuh's K-major no-swizzle layout (dgemm.cu's, whose gate is exact on both cards).
// Build: nvcc -O2 -std=c++17 -gencode arch=compute_90a,code=sm_90a s8_mma.cu -o s8_mma_sm90 (or compute_100a / sm_100a)
// Each form is its own extern "C" kernel, so cuobjdump -sass shows each one's tensor instruction by name.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "../dc.cuh"
#include "../wgmma.cuh"

// ---- mma.sync m16n8k32 (benchmarks/tc_probe/mma_tiles.cu's 8-bit fragment layout) ---------------------------------------
#define MMA_FORM(NAME, PTX)                                                                                               \
  extern "C" __global__ void NAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,                          \
                                  const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int ntiles) {                 \
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                                        \
    if (warp >= ntiles) return;                                                                                           \
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;                                                       \
    const uint8_t* a = A + (size_t)warp * 512;                                                                            \
    const uint8_t* b = Bt + (size_t)warp * 256;                                                                           \
    const uint32_t* c = C + (size_t)warp * 128;                                                                           \
    uint32_t* d = D + (size_t)warp * 128;                                                                                 \
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(a + g * 32 + 4 * t);                                           \
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(a + (g + 8) * 32 + 4 * t);                                     \
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(a + g * 32 + 16 + 4 * t);                                      \
    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(a + (g + 8) * 32 + 16 + 4 * t);                                \
    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b + g * 32 + 4 * t);                                           \
    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(b + g * 32 + 16 + 4 * t);                                      \
    const uint32_t c0 = c[g * 8 + 2 * t], c1 = c[g * 8 + 2 * t + 1];                                                      \
    const uint32_t c2 = c[(g + 8) * 8 + 2 * t], c3 = c[(g + 8) * 8 + 2 * t + 1];                                          \
    uint32_t d0, d1, d2, d3;                                                                                              \
    asm volatile(PTX " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                      \
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)                                                                 \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));             \
    d[g * 8 + 2 * t] = d0;                                                                                                \
    d[g * 8 + 2 * t + 1] = d1;                                                                                            \
    d[(g + 8) * 8 + 2 * t] = d2;                                                                                          \
    d[(g + 8) * 8 + 2 * t + 1] = d3;                                                                                      \
  }

MMA_FORM(mma_s8s8_sat, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32")
MMA_FORM(mma_s8u8_sat, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.u8.s32")
MMA_FORM(mma_u8s8_sat, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32")
MMA_FORM(mma_u8u8_sat, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32")
MMA_FORM(mma_s8s8_wrap, "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
MMA_FORM(mma_s8u8_wrap, "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32")
MMA_FORM(mma_u8s8_wrap, "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32")
MMA_FORM(mma_u8u8_wrap, "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32")

// Copies a rows x 32-byte tile (k contiguous) into dc.cuh's K-major layout. All threads of the block.
__device__ __forceinline__ void load_kmaj(unsigned char* dst, const uint8_t* src, int rows) {
  for (int c = threadIdx.x; c < rows * 2; c += blockDim.x) {
    const int r = c >> 1, kc = c & 1;
    *reinterpret_cast<uint4*>(dst + kmaj_off(r, 16 * kc, 32)) = *reinterpret_cast<const uint4*>(src + (size_t)r * 32 + 16 * kc);
  }
}

// ---- sm_90a: wgmma m64n8k32 -------------------------------------------------------------------------------------------
// Thread (warp w, lane = 4 g + q) holds D[16 w + g + 8 (e >> 1)][2 q + (e & 1)] in d[e] (wgmma's D fragment, dgemm.cu's).
#define WG_FORM(NAME, PTX)                                                                                                \
  extern "C" __global__ void __launch_bounds__(128) NAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,   \
                                                         const uint32_t* __restrict__ C, uint32_t* __restrict__ D,        \
                                                         int ntiles) {                                                    \
    WG_BODY(PTX)                                                                                                          \
  }
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define WG_BODY(PTX)                                                                                                      \
  __shared__ __align__(1024) unsigned char sA[64 * 32];                                                                   \
  __shared__ __align__(1024) unsigned char sB[8 * 32];                                                                    \
  const int tile = blockIdx.x;                                                                                            \
  if (tile >= ntiles) return;                                                                                             \
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;                                   \
  load_kmaj(sA, A + (size_t)tile * 64 * 32, 64);                                                                          \
  load_kmaj(sB, Bt + (size_t)tile * 8 * 32, 8);                                                                           \
  fence_async_smem();                                                                                                     \
  __syncthreads();                                                                                                        \
  const uint32_t* c = C + (size_t)tile * 512;                                                                             \
  uint32_t* d = D + (size_t)tile * 512;                                                                                   \
  uint32_t r[4];                                                                                                          \
  _Pragma("unroll") for (int e = 0; e < 4; e++) r[e] = c[(16 * w + g + 8 * (e >> 1)) * 8 + 2 * q + (e & 1)];            \
  const uint64_t da = wgmma_desc(smem_u32(sA), kmaj_lbo(), kmaj_sbo(32));                                                 \
  const uint64_t db = wgmma_desc(smem_u32(sB), kmaj_lbo(), kmaj_sbo(32));                                                 \
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");                                                                 \
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n" PTX " {%0, %1, %2, %3}, %4, %5, p;\n}\n"                       \
               : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3])                                                           \
               : "l"(da), "l"(db), "r"(1));                                                                               \
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");                                                          \
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");                                                          \
  _Pragma("unroll") for (int e = 0; e < 4; e++) asm volatile("" : "+r"(r[e])::"memory");                                 \
  _Pragma("unroll") for (int e = 0; e < 4; e++) d[(16 * w + g + 8 * (e >> 1)) * 8 + 2 * q + (e & 1)] = r[e];
#else
#define WG_BODY(PTX)
#endif

WG_FORM(wg_s8s8_sat, "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite")
WG_FORM(wg_s8u8_sat, "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8.satfinite")
WG_FORM(wg_u8s8_sat, "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8.satfinite")
WG_FORM(wg_u8u8_sat, "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8.satfinite")
WG_FORM(wg_s8s8_wrap, "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8")
WG_FORM(wg_s8u8_wrap, "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8")
WG_FORM(wg_u8s8_wrap, "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8")
WG_FORM(wg_u8u8_wrap, "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8")

// ---- sm_90a: wgmma m64nNk32, N = 64, 128, 256 -------------------------------------------------------------------------
// Thread (warp w, lane = 4 g + q) holds D[16 w + g + 8 ((e >> 1) & 1)][8 (e >> 2) + 2 q + (e & 1)] in d[e], e < N / 2:
// m64n8k32's fragment once for each 8 columns.
#define WGN_FORM(NAME, N, FN)                                                                                             \
  extern "C" __global__ void __launch_bounds__(128) NAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,   \
                                                         const uint32_t* __restrict__ C, uint32_t* __restrict__ D,        \
                                                         int ntiles) {                                                    \
    WGN_BODY(N, FN)                                                                                                       \
  }
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define WGN_AT(N, e) ((16 * w + g + 8 * (((e) >> 1) & 1)) * (N) + 8 * ((e) >> 2) + 2 * q + ((e) & 1))
#define WGN_BODY(N, FN)                                                                                                   \
  __shared__ __align__(1024) unsigned char sA[64 * 32];                                                                   \
  __shared__ __align__(1024) unsigned char sB[N * 32];                                                                    \
  const int tile = blockIdx.x;                                                                                            \
  if (tile >= ntiles) return;                                                                                             \
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;                                   \
  load_kmaj(sA, A + (size_t)tile * 64 * 32, 64);                                                                          \
  load_kmaj(sB, Bt + (size_t)tile * N * 32, N);                                                                           \
  fence_async_smem();                                                                                                     \
  __syncthreads();                                                                                                        \
  const uint32_t* c = C + (size_t)tile * 64 * N;                                                                          \
  uint32_t* d = D + (size_t)tile * 64 * N;                                                                                \
  int r[N / 2];                                                                                                           \
  _Pragma("unroll") for (int e = 0; e < N / 2; e++) r[e] = (int)c[WGN_AT(N, e)];                                         \
  wgmma_hold(r);                                                                                                          \
  wgmma_fence();                                                                                                          \
  FN(r, wgmma_desc(smem_u32(sA), kmaj_lbo(), kmaj_sbo(32)), wgmma_desc(smem_u32(sB), kmaj_lbo(), kmaj_sbo(32)), 1);       \
  wgmma_commit();                                                                                                         \
  wgmma_wait<0>();                                                                                                        \
  wgmma_hold(r);                                                                                                          \
  _Pragma("unroll") for (int e = 0; e < N / 2; e++) d[WGN_AT(N, e)] = (uint32_t)r[e];
#else
#define WGN_BODY(N, FN)
#endif
#define WGN_FORMS(N)                                                                                                      \
  WGN_FORM(wg##N##_s8s8_sat, N, wgmma_i8_s8s8_sat_n##N)                                                                   \
  WGN_FORM(wg##N##_s8u8_sat, N, wgmma_i8_s8u8_sat_n##N)                                                                   \
  WGN_FORM(wg##N##_u8s8_sat, N, wgmma_i8_u8s8_sat_n##N)                                                                   \
  WGN_FORM(wg##N##_u8u8_sat, N, wgmma_i8_u8u8_sat_n##N)                                                                   \
  WGN_FORM(wg##N##_s8s8_wrap, N, wgmma_i8_s8s8_wrap_n##N)                                                                 \
  WGN_FORM(wg##N##_s8u8_wrap, N, wgmma_i8_s8u8_wrap_n##N)                                                                 \
  WGN_FORM(wg##N##_u8s8_wrap, N, wgmma_i8_u8s8_wrap_n##N)                                                                 \
  WGN_FORM(wg##N##_u8u8_wrap, N, wgmma_i8_u8u8_wrap_n##N)
WGN_FORMS(64)
WGN_FORMS(128)
WGN_FORMS(256)

// ---- sm_100a: tcgen05.mma kind::i8, M 128, N 32 ---------------------------------------------------------------------------
// D row i is TMEM lane i, column j TMEM column j; warp w reaches lanes 32 w .. 32 w + 31, so thread (w, lane) owns row
// 32 w + lane: it stores C's 32 words of that row into TMEM before the MMA and loads D's after it.
template <uint32_t AFMT, uint32_t BFMT, uint32_t SAT>
__device__ __forceinline__ void tc_tile(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,
                                        const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int ntiles) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  constexpr int M = 128, N = 32;
  __shared__ __align__(1024) unsigned char sA[M * 32];
  __shared__ __align__(1024) unsigned char sB[N * 32];
  __shared__ uint64_t bar;
  __shared__ uint32_t tslot;
  const int tile = blockIdx.x;
  if (tile >= ntiles) return;
  const uint32_t w = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
  const int row = 32 * w + lane;
  if (threadIdx.x == 0) {
    mbar_init(&bar, 1);
    fence_mbar_init();
  }
  if (w == 0) tmem_alloc(&tslot, N);
  load_kmaj(sA, A + (size_t)tile * M * 32, M);
  load_kmaj(sB, Bt + (size_t)tile * N * 32, N);
  fence_async_smem();
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = tslot, trow = tmem + ((32u * w) << 16);
  const uint32_t* crow = C + ((size_t)tile * M + row) * N;
  uint32_t r[16];
#pragma unroll
  for (int h = 0; h < 2; h++) {
#pragma unroll
    for (int j = 0; j < 16; j++) r[j] = crow[16 * h + j];
    tmem_st16(trow + 16 * h, r);
  }
  tmem_wait_st();
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (w == 0) {
    if (elect_one()) {
      constexpr uint32_t idesc = umma_idesc(2, AFMT, BFMT, M, N) | (SAT << 3);
      umma_i8(tmem, umma_desc(smem_u32(sA), kmaj_lbo(), kmaj_sbo(32)), umma_desc(smem_u32(sB), kmaj_lbo(), kmaj_sbo(32)),
              idesc, 1);
      umma_commit(&bar);
    }
    __syncwarp();
  }
  mbar_wait(&bar, 0);
  tc_after_sync();
  uint32_t* drow = D + ((size_t)tile * M + row) * N;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    tmem_ld16(trow + 16 * h, r);
    tmem_wait_ld();
#pragma unroll
    for (int j = 0; j < 16; j++) drow[16 * h + j] = r[j];
  }
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (w == 0) tmem_dealloc(tmem, N);
#endif
}

// kind::i8 formats: u8 0, s8 1
#define TC_FORM(NAME, AF, BF, SAT)                                                                                        \
  extern "C" __global__ void __launch_bounds__(128) NAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,   \
                                                         const uint32_t* __restrict__ C, uint32_t* __restrict__ D,        \
                                                         int ntiles) {                                                    \
    tc_tile<AF, BF, SAT>(A, Bt, C, D, ntiles);                                                                            \
  }
TC_FORM(tc_s8s8_sat, 1, 1, 1)
TC_FORM(tc_s8u8_sat, 1, 0, 1)
TC_FORM(tc_u8s8_sat, 0, 1, 1)
TC_FORM(tc_u8u8_sat, 0, 0, 1)
TC_FORM(tc_s8s8_wrap, 1, 1, 0)
TC_FORM(tc_s8u8_wrap, 1, 0, 0)
TC_FORM(tc_u8s8_wrap, 0, 1, 0)
TC_FORM(tc_u8u8_wrap, 0, 0, 0)

namespace {

typedef void (*kernel_fn)(const uint8_t*, const uint8_t*, const uint32_t*, uint32_t*, int);

struct Form {
  const char* name;
  kernel_fn kernel;
};
struct Family {
  const char* name;
  int m, n, threads, tiles_per_block;
  Form forms[8];
};

const Family FAMILIES[] = {
    {"mma", 16, 8, 128, 4,
     {{"s8s8_sat", mma_s8s8_sat}, {"s8u8_sat", mma_s8u8_sat}, {"u8s8_sat", mma_u8s8_sat}, {"u8u8_sat", mma_u8u8_sat},
      {"s8s8_wrap", mma_s8s8_wrap}, {"s8u8_wrap", mma_s8u8_wrap}, {"u8s8_wrap", mma_u8s8_wrap}, {"u8u8_wrap", mma_u8u8_wrap}}},
    {"wg", 64, 8, 128, 1,
     {{"s8s8_sat", wg_s8s8_sat}, {"s8u8_sat", wg_s8u8_sat}, {"u8s8_sat", wg_u8s8_sat}, {"u8u8_sat", wg_u8u8_sat},
      {"s8s8_wrap", wg_s8s8_wrap}, {"s8u8_wrap", wg_s8u8_wrap}, {"u8s8_wrap", wg_u8s8_wrap}, {"u8u8_wrap", wg_u8u8_wrap}}},
#define WGN_FAMILY(N)                                                                                                     \
  {"wg" #N, 64, N, 128, 1,                                                                                                \
   {{"s8s8_sat", wg##N##_s8s8_sat}, {"s8u8_sat", wg##N##_s8u8_sat}, {"u8s8_sat", wg##N##_u8s8_sat},                       \
    {"u8u8_sat", wg##N##_u8u8_sat}, {"s8s8_wrap", wg##N##_s8s8_wrap}, {"s8u8_wrap", wg##N##_s8u8_wrap},                   \
    {"u8s8_wrap", wg##N##_u8s8_wrap}, {"u8u8_wrap", wg##N##_u8u8_wrap}}}
    WGN_FAMILY(64), WGN_FAMILY(128), WGN_FAMILY(256),
    {"tc", 128, 32, 128, 1,
     {{"s8s8_sat", tc_s8s8_sat}, {"s8u8_sat", tc_s8u8_sat}, {"u8s8_sat", tc_u8s8_sat}, {"u8u8_sat", tc_u8u8_sat},
      {"s8s8_wrap", tc_s8s8_wrap}, {"s8u8_wrap", tc_s8u8_wrap}, {"u8s8_wrap", tc_u8s8_wrap}, {"u8u8_wrap", tc_u8u8_wrap}}},
};

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    exit(3);
  }
}

std::vector<uint8_t> read_all(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(2);
  }
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(n);
  if (fread(buf.data(), 1, n, f) != (size_t)n) {
    fprintf(stderr, "short read of %s\n", path);
    exit(2);
  }
  fclose(f);
  return buf;
}

void write_all(const std::string& path, const void* data, size_t n) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(data, 1, n, f) != n) {
    fprintf(stderr, "cannot write %s\n", path.c_str());
    exit(2);
  }
  fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: s8_mma mma|wg|wg64|wg128|wg256|tc CASES OUTDIR\n");
    return 2;
  }
  const Family* fam = nullptr;
  for (const Family& f : FAMILIES)
    if (strcmp(f.name, argv[1]) == 0) fam = &f;
  if (!fam) {
    fprintf(stderr, "no family %s (mma, wg, wg64, wg128, wg256, tc)\n", argv[1]);
    return 2;
  }
  const std::vector<uint8_t> cases = read_all(argv[2]);
  if (cases.size() < 16 || memcmp(cases.data(), "S8MMAPR1", 8) != 0) {
    fprintf(stderr, "%s is not an s8_mma case file\n", argv[2]);
    return 2;
  }
  uint32_t ntiles;
  uint16_t m, n;
  memcpy(&ntiles, cases.data() + 8, 4);
  memcpy(&m, cases.data() + 12, 2);
  memcpy(&n, cases.data() + 14, 2);
  if (m != fam->m || n != fam->n) {
    fprintf(stderr, "%s holds %u x %u tiles; family %s takes %d x %d\n", argv[2], m, n, fam->name, fam->m, fam->n);
    return 2;
  }
  const size_t a_bytes = (size_t)ntiles * m * 32, b_bytes = (size_t)ntiles * n * 32, c_bytes = (size_t)ntiles * m * n * 4;
  if (cases.size() != 16 + a_bytes + b_bytes + c_bytes) {
    fprintf(stderr, "%s: size %zu does not fit %u tiles\n", argv[2], cases.size(), ntiles);
    return 2;
  }
  const std::string out = argv[3];

  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  char ident[512];
  snprintf(ident, sizeof ident, "name=%s\ncc=%d.%d\nsms=%d\ndriver_api=%d\nruntime=%d\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, driver, runtime);
  write_all(out + "/device.txt", ident, strlen(ident));

  uint8_t *dA, *dB;
  uint32_t *dC, *dD;
  check(cudaMalloc(&dA, a_bytes), "cudaMalloc A");
  check(cudaMalloc(&dB, b_bytes), "cudaMalloc B");
  check(cudaMalloc(&dC, c_bytes), "cudaMalloc C");
  check(cudaMalloc(&dD, c_bytes), "cudaMalloc D");
  check(cudaMemcpy(dA, cases.data() + 16, a_bytes, cudaMemcpyHostToDevice), "copy A");
  check(cudaMemcpy(dB, cases.data() + 16 + a_bytes, b_bytes, cudaMemcpyHostToDevice), "copy B");
  check(cudaMemcpy(dC, cases.data() + 16 + a_bytes + b_bytes, c_bytes, cudaMemcpyHostToDevice), "copy C");
  std::vector<uint32_t> host((size_t)ntiles * m * n);
  const int blocks = (int)((ntiles + fam->tiles_per_block - 1) / fam->tiles_per_block);
  for (const Form& form : fam->forms) {
    check(cudaMemset(dD, 0xA5, c_bytes), "memset D");  // a word the kernel never wrote reads 0xA5A5A5A5
    form.kernel<<<blocks, fam->threads>>>(dA, dB, dC, dD, (int)ntiles);
    check(cudaGetLastError(), form.name);
    check(cudaDeviceSynchronize(), form.name);
    check(cudaMemcpy(host.data(), dD, c_bytes, cudaMemcpyDeviceToHost), "copy D");
    write_all(out + "/d_" + form.name + ".bin", host.data(), c_bytes);
    printf("%s_%s: %u tiles\n", fam->name, form.name, ntiles);
  }
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dD);
  return 0;
}
