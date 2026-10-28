// P2 decode into a staging buffer: one tensor of W's plaintext bytes, decoded from the P2 store C (`p2-16448/v4`,
// verity/core/protocols/pous/schemes/p2.py) for the exact INT8 GEMM to read as its weight operand ([N, K] row-major, as
// W holds it). Built as a shared library for ctypes (stage.py).
//
// Reused by include, unmodified (memory-accounting's, benchmarks/pous/p2_v1):
//   p2sq.cuh     p2sq::decode_one, m = sq(sigma_K(sq(c))) - t mod p of one block a warp on its Karatsuba squaring
//                (P2SQ_KARA, the Makefile's default), K and t as big-endian rows; p2sq::prefetch_row
//   p2keys4.cuh  p2keys4::midstate and p2keys4::chunk, v4's counter-mode SHA-512 key chunks after the prefix's midstate
//
// Block j of the store (its global index: segment j / per_segment, local block j % per_segment) carries W bytes
// [2054 j, 2054 j + 2054) as bytes 2 .. 2055 of its big-endian 2,056-byte row. A call decodes blocks first .. first + n - 1
// (c points at block first's row) and writes W bytes [base, base + len) to dst[0 .. len): the tensor at W-offset base,
// clipped at both ends.
//
// p2stage_decode: each warp takes a contiguous run of the call's blocks and, for each 32 of them, first expands their K and t
// rows (a lane a block, 33 chunks each) into its own scratch rows, then decodes them one by one, so the warps of a scheduler
// mix key expansion (ALU) with squaring (IMAD). p2stage_decode_keys takes K and t rows from the caller instead (the
// library's keys, as gpu.Kernels.keys gives them), the check of the expansion.
//
// flags, OR'd on the device and never fixed silently: bit 0 a t whose first candidate is >= p (decode_one's; the spec
// draws the next candidate), bit 2 a block whose decode has more than 16,432 bits (not a codeword: the registered
// reference refuses it), bit 3 a c >= p (refused there too). Payload bytes are written either way, so a check can compare
// them with the reference's on any canonical c.
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>

#ifndef P2_BITS
#define P2_BITS 16448
#endif
#if P2_BITS != 16448
#error "p2sq.cuh decodes at 16,448 bits"
#endif
#ifndef P2ST_SCHOOLBOOK
#define P2SQ_KARA
#endif
#include "../../pous/p2_v1/p2sq.cuh"
#include "../../pous/p2_v1/p2keys4.cuh"

#ifndef P2ST_WARPS
#define P2ST_WARPS 2
#endif
#ifndef P2ST_MINB
#define P2ST_MINB 1
#endif

namespace p2st {

constexpr int N = p2sq::N;                       // 514 words a row
constexpr int PAYLOAD = 4 * N - 2;               // 2,054 bytes of W a block
constexpr int ROW64 = p2keys4::ROW_WORDS;        // 257 eight-byte words a row
constexpr uint32_t BATCH = 32;
constexpr uint64_t KEYED_WORDS = 2 * BATCH * ROW64 + ROW64;   // a warp's K and t rows of a batch, then its m row
constexpr uint64_t PLAIN_WORDS = ROW64;                       // a warp's m row
static_assert(N == 2 * ROW64, "a row is 257 eight-byte words");

// rows (<= 32) blocks' K and t rows, a lane a block
__device__ __forceinline__ void keys_batch(const p2keys4::Mid &mid, uint64_t first, uint32_t rows, uint64_t per_segment,
                                           uint64_t *__restrict__ K, uint64_t *__restrict__ T, uint32_t lane) {
  if (lane >= rows) return;
  const uint64_t idx = first + lane, seg = idx / per_segment, blk = idx % per_segment;
  uint64_t *kr = K + (size_t)lane * ROW64, *tr = T + (size_t)lane * ROW64;
#pragma unroll 1
  for (uint32_t c = 0; c < (uint32_t)p2keys4::CHUNKS; c++) {
    uint64_t o[8];
    p2keys4::chunk(mid, seg, blk, p2keys4::w4(0, c), o);
#pragma unroll
    for (int q = 0; q < 8; q++)
      if (8 * c + q < (uint32_t)ROW64) kr[8 * c + q] = p2keys4::bswap(o[q]);
    p2keys4::chunk(mid, seg, blk, p2keys4::w4(1, c), o);
#pragma unroll
    for (int q = 0; q < 8; q++)
      if (8 * c + q < (uint32_t)ROW64) tr[8 * c + q] = p2keys4::bswap(o[q]);
  }
}

// the big-endian row's value >= p = 2^W - c0: every word but the last all ones, and the last >= 2^32 - c0
__device__ __forceinline__ bool row_ge_p(const uint32_t *__restrict__ row, uint32_t c0, uint32_t lane) {
  uint32_t ones = 0xFFFFFFFFu;
  bool last = true;
  for (int w = (int)lane; w < N; w += 32) {
    const uint32_t v = __byte_perm(row[w], 0, 0x0123);
    if (w == N - 1) last = v >= 0u - c0;
    else ones &= v;
  }
  return __all_sync(0xFFFFFFFFu, ones == 0xFFFFFFFFu && last);
}

// the block's payload (row bytes 2 .. 2055) to dst[at .. at + 2054), clipped to [0, len); true on every lane when the
// decode is wider than the payload (row bytes 0 and 1 not zero)
__device__ __forceinline__ bool payload_out(const uint32_t *__restrict__ mrow, uint8_t *__restrict__ dst, int64_t at, int64_t len,
                                            uint32_t lane) {
  uint32_t v[17];
#pragma unroll
  for (int i = 0; i < 17; i++) {
    const int w = (int)lane + 32 * i;
    v[i] = w < N ? mrow[w] : 0;
  }
  const bool wide = (__shfl_sync(0xFFFFFFFFu, v[0], 0) & 0xFFFFu) != 0;
#pragma unroll
  for (int i = 0; i < 17; i++) {
    const int w = (int)lane + 32 * i;
    if (w < N) {
#pragma unroll
      for (int b = 0; b < 4; b++) {
        const int64_t q = 4 * w + b - 2, d = at + q;
        if (q >= 0 && d >= 0 && d < len) dst[d] = (uint8_t)(v[i] >> (8 * b));
      }
    }
  }
  return wide;
}

__device__ __forceinline__ void record_sm(uint32_t *smids) {
  if (smids != nullptr && threadIdx.x == 0) {
    uint32_t s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    smids[blockIdx.x] = s;
  }
}

}  // namespace p2st

__global__ void __launch_bounds__(32 * P2ST_WARPS, P2ST_MINB) p2st_keyed_kernel(
    const __grid_constant__ p2keys4::Mid mid, const uint32_t *__restrict__ cin, uint64_t first, uint32_t n, uint64_t per_segment,
    uint32_t c0, uint8_t *__restrict__ dst, int64_t base, int64_t len, uint32_t *__restrict__ flags, uint64_t *__restrict__ scratch,
    uint32_t *__restrict__ smids) {
  using namespace p2st;
  __shared__ __align__(16) uint32_t smem[P2ST_WARPS][p2sq::WARP_WORDS];
  record_sm(smids);
  const uint32_t lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  uint32_t *sm = smem[wid], *snap = smem[wid] + p2sq::E_WORDS;
  const uint64_t warps = (uint64_t)gridDim.x * P2ST_WARPS, w = (uint64_t)blockIdx.x * P2ST_WARPS + wid;
  const uint32_t lo = (uint32_t)(n * w / warps), hi = (uint32_t)(n * (w + 1) / warps);
  uint64_t *K = scratch + w * KEYED_WORDS, *T = K + BATCH * ROW64;
  uint32_t *mrow = (uint32_t *)(T + BATCH * ROW64);
  uint32_t bad = 0;
  for (uint32_t j0 = lo; j0 < hi; j0 += BATCH) {
    const uint32_t rows = hi - j0 < BATCH ? hi - j0 : BATCH;
    __syncwarp();
    keys_batch(mid, first + j0, rows, per_segment, K, T, lane);
    __syncwarp();
    for (uint32_t r = 0; r < rows; r++) {
      const uint32_t j = j0 + r;
      const uint32_t *crow = cin + (size_t)j * N;
      if (j + 1 < hi) p2sq::prefetch_row(crow + N, lane);
      if (row_ge_p(crow, c0, lane)) bad |= 8u;
      p2sq::decode_one(sm, snap, crow, mrow, (const uint32_t *)(K + (size_t)r * ROW64), (const uint32_t *)(T + (size_t)r * ROW64),
                       c0, flags, lane);
      __syncwarp();
      if (payload_out(mrow, dst, (int64_t)PAYLOAD * (int64_t)(first + j) - base, len, lane)) bad |= 4u;
      __syncwarp();
    }
  }
  if (bad != 0 && lane == 0) atomicOr(flags, bad);
}

__global__ void __launch_bounds__(32 * P2ST_WARPS, P2ST_MINB) p2st_plain_kernel(
    const uint32_t *__restrict__ cin, const uint32_t *__restrict__ kin, const uint32_t *__restrict__ tin, uint64_t first, uint32_t n,
    uint32_t c0, uint8_t *__restrict__ dst, int64_t base, int64_t len, uint32_t *__restrict__ flags, uint64_t *__restrict__ scratch,
    uint32_t *__restrict__ smids) {
  using namespace p2st;
  __shared__ __align__(16) uint32_t smem[P2ST_WARPS][p2sq::WARP_WORDS];
  record_sm(smids);
  const uint32_t lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  uint32_t *sm = smem[wid], *snap = smem[wid] + p2sq::E_WORDS;
  const uint64_t warps = (uint64_t)gridDim.x * P2ST_WARPS, w = (uint64_t)blockIdx.x * P2ST_WARPS + wid;
  const uint32_t lo = (uint32_t)(n * w / warps), hi = (uint32_t)(n * (w + 1) / warps);
  uint32_t *mrow = (uint32_t *)(scratch + w * PLAIN_WORDS);
  uint32_t bad = 0;
  for (uint32_t j = lo; j < hi; j++) {
    const uint32_t *crow = cin + (size_t)j * N;
    if (j + 1 < hi) p2sq::prefetch_row(crow + N, lane);
    if (row_ge_p(crow, c0, lane)) bad |= 8u;
    p2sq::decode_one(sm, snap, crow, mrow, kin + (size_t)j * N, tin + (size_t)j * N, c0, flags, lane);
    __syncwarp();
    if (payload_out(mrow, dst, (int64_t)PAYLOAD * (int64_t)(first + j) - base, len, lane)) bad |= 4u;
    __syncwarp();
  }
  if (bad != 0 && lane == 0) atomicOr(flags, bad);
}

namespace {

int sms() {
  int dev = 0, n = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  return n > 0 ? n : 1;
}

uint32_t default_grid(const void *kernel, uint32_t n) {
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, 32 * P2ST_WARPS, 0);
  const uint64_t full = ((uint64_t)n + P2ST_WARPS - 1) / P2ST_WARPS;
  return (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(full, (uint64_t)std::max(occ, 1) * sms()));
}

const void *kernel_of(int keyed) {
  return keyed ? (const void *)p2st_keyed_kernel : (const void *)p2st_plain_kernel;
}

}  // namespace

extern "C" int p2stage_version() { return 1; }
extern "C" int p2stage_width_bits() { return P2_BITS; }
extern "C" int p2stage_warps_per_cta() { return P2ST_WARPS; }

// the grid a call of n blocks launches when it names none (resident CTAs an SM times the SMs, at most a warp a block)
extern "C" uint32_t p2stage_grid(int keyed, uint32_t n) { return default_grid(kernel_of(keyed), n); }

// the scratch a launch of `grid` CTAs needs, in bytes
extern "C" uint64_t p2stage_scratch_bytes(int keyed, uint32_t grid) {
  return (uint64_t)grid * P2ST_WARPS * (keyed ? p2st::KEYED_WORDS : p2st::PLAIN_WORDS) * 8;
}

// registers a thread, static shared memory a CTA, resident CTAs an SM: the budget co-residency with a GEMM works from
extern "C" int p2stage_attributes(int keyed, int *regs, int *smem, int *ctas_per_sm) {
  cudaFuncAttributes a;
  cudaError_t e = cudaFuncGetAttributes(&a, kernel_of(keyed));
  if (e != cudaSuccess) return (int)e;
  *regs = a.numRegs;
  *smem = (int)a.sharedSizeBytes;
  return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(ctas_per_sm, kernel_of(keyed), 32 * P2ST_WARPS, 0);
}

// keys expanded in the kernel, under the expansion prefix (whole 128-byte SHA-512 blocks); -1 a prefix that isn't,
// -2 too little scratch
extern "C" int p2stage_decode(const void *c, uint64_t first, uint32_t n, const uint8_t *prefix, int plen, uint64_t per_segment,
                              uint32_t c0, void *dst, int64_t base, int64_t len, void *flags, void *scratch, uint64_t scratch_bytes,
                              uint32_t grid, void *smids, void *stream) {
  p2keys4::Mid mid;
  if (!p2keys4::midstate(prefix, plen, mid)) return -1;
  if (n == 0) return 0;
  if (grid == 0) grid = default_grid(kernel_of(1), n);
  if (scratch_bytes < p2stage_scratch_bytes(1, grid)) return -2;
  p2st_keyed_kernel<<<grid, 32 * P2ST_WARPS, 0, (cudaStream_t)stream>>>(mid, (const uint32_t *)c, first, n, per_segment, c0,
                                                                         (uint8_t *)dst, base, len, (uint32_t *)flags,
                                                                         (uint64_t *)scratch, (uint32_t *)smids);
  return (int)cudaGetLastError();
}

// K and t as rows (gpu.Kernels.keys' [n, W/8] big-endian rows of blocks first .. first + n - 1); -2 too little scratch
extern "C" int p2stage_decode_keys(const void *c, const void *k, const void *t, uint64_t first, uint32_t n, uint32_t c0, void *dst,
                                   int64_t base, int64_t len, void *flags, void *scratch, uint64_t scratch_bytes, uint32_t grid,
                                   void *smids, void *stream) {
  if (n == 0) return 0;
  if (grid == 0) grid = default_grid(kernel_of(0), n);
  if (scratch_bytes < p2stage_scratch_bytes(0, grid)) return -2;
  p2st_plain_kernel<<<grid, 32 * P2ST_WARPS, 0, (cudaStream_t)stream>>>((const uint32_t *)c, (const uint32_t *)k, (const uint32_t *)t,
                                                                         first, n, c0, (uint8_t *)dst, base, len, (uint32_t *)flags,
                                                                         (uint64_t *)scratch, (uint32_t *)smids);
  return (int)cudaGetLastError();
}
