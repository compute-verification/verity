// Single-instruction tensor-core probe for datacenter Blackwell (sm_100a): ONE `tcgen05.mma.cta_group::1.kind::f16` with BF16
// operands per tile, the accumulator in tensor memory, raw bits in and out.
//
//   tcgen05_bf16_m128n8k16_tiles         one tcgen05.mma M128 N16 K16 per tile        A 128x16 bf16, Bt 8x16 bf16, C/D 128x8 f32
//   tcgen05_bf16_m128n8k16_chain4_tiles  four back to back on one TMEM accumulator     A 128x64 bf16, Bt 8x64 bf16, C/D 128x8 f32
//
// This is the k16 step cuBLAS's sm_100 BF16 GEMMs issue (UTCHMMA).  After benchmarks/tc_probe_fp4/tcgen05_fp4.cu (the
// block-scaled FP4 kinds on the same die), with the tile interface of wgmma_tiles.cu so tc_probe.py's families, layout check and
// models apply unchanged.  Per tile (one 128-thread CTA, thread m owns row m):
//   1. warp 0 allocates 32 TMEM columns (tcgen05.alloc); thread 0 initialises an mbarrier.
//   2. A (128 x K) and B (16 x K: the tile's 8 columns twice) go to shared memory in the canonical K-major no-swizzle layout the
//      shared-memory descriptor describes: core matrices of 8 rows x 16 bytes (8 bf16), the second 16-byte K chunk of a row group
//      at `lbo`, the next 8-row group at `sbo` (runtime, so the host's exact-integer layout check can try the swapped pair).
//   3. each warp preloads its 32 TMEM lanes: lane m, columns 0..15 = C[m][n % 8] (tcgen05.st).
//   4. thread 0 issues the MMA(s) -- `tcgen05.mma.cta_group::1.kind::f16 [d], a_desc, b_desc, idesc, enable_input_d` with the
//      instruction descriptor BF16 x BF16 -> F32, K-major A and B, N = 16, M = 128 -- then `tcgen05.commit` onto the mbarrier.  The
//      first MMA's enable_input_d is `tcgen05_set_enable_input_d`'s (1: D = A.B + C; 0: D = A.B, C not read, as a GEMM mainloop's
//      first k step), every later one's 1.  `#pragma unroll 1` keeps ONE tcgen05.mma instruction in the SASS whatever the step count.
//   5. every thread waits on the mbarrier, its warp reads its 32 lanes x 16 columns with tcgen05.ld; columns 0..7 are D, columns
//      8..15 a replica the host compares (`tcgen05_replica_mismatches`): a word that differs between the two halves of the
//      N = 16 instruction is counted, never silently dropped.
// N = 16 because the PTX ISA's kind::f16 shapes at M = 128 step N by 16.
//
// Descriptors: PTX ISA "tcgen05 shared memory descriptor" (start >> 4 at 0-13, LBO >> 4 at 16-29, SBO >> 4 at 32-45, the fixed
// 0b001 at 46-48, base offset 0, no swizzle) and "instruction descriptor for .kind::f16" (D format F32 = 1 at bits 4-5, A/B
// format BF16 = 1 at 7-9 / 10-12, K-major A/B (bits 15/16 = 0), N >> 3 at 17-22, M >> 4 at 24-28), as CUTLASS 3.9's
// UMMA::SmemDescriptor and UMMA::InstrDescriptor lay them out.
//
// Build: nvcc -O2 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared -o libtcgen05_tiles_sm_100a.so tcgen05_tiles.cu
// (the arch-specific target: tcgen05 is an `a`-feature instruction).

#include "tiles_common.cuh"

namespace {

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM100_ALL)
#define TC_PROBE_TCGEN05_ENABLED 1
#else
#define TC_PROBE_TCGEN05_ENABLED 0
#endif

constexpr int M = 128;
constexpr int N = 8;       // the tile's columns
constexpr int MMA_N = 16;  // the instruction's: columns 8..15 replicate 0..7
constexpr int K16 = 16;
constexpr int THREADS = 128;
constexpr int MAX_STEPS = 4;
constexpr int A_STEP_BYTES = M * K16 * 2;      // 4096
constexpr int B_STEP_BYTES = MMA_N * K16 * 2;  // 512
constexpr int TMEM_COLS = 32;

uint32_t g_lbo = 128, g_sbo = 256, g_enable_input_d = 1;
__device__ unsigned long long g_replica_mismatches;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFFu) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFFu) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFFu) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    return d;
}

__host__ __device__ constexpr uint32_t make_idesc() {
    return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(MMA_N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

// Row r's K16 slice s (16 bf16 words) into a step block: chunk c = k / 8 at r/8 * sbo + c * lbo + r%8 * 16.
__device__ __forceinline__ void store_row(uint8_t* blk, int r, const uint16_t* src, uint32_t lbo, uint32_t sbo) {
#pragma unroll
    for (int c = 0; c < 2; c++) {
        const uint16_t* w = src + 8 * c;
        const uint4 v = make_uint4((uint32_t)w[0] | ((uint32_t)w[1] << 16), (uint32_t)w[2] | ((uint32_t)w[3] << 16),
                                   (uint32_t)w[4] | ((uint32_t)w[5] << 16), (uint32_t)w[6] | ((uint32_t)w[7] << 16));
        *reinterpret_cast<uint4*>(blk + (r / 8) * sbo + c * lbo + (r % 8) * 16) = v;
    }
}

template <int STEPS>
__device__ __forceinline__ void tile_body(const uint16_t* __restrict__ A, const uint16_t* __restrict__ Bt, const uint32_t* __restrict__ C,
                                          uint32_t* __restrict__ D, int ntiles, uint32_t lbo, uint32_t sbo, uint32_t enable_first) {
#if TC_PROBE_TCGEN05_ENABLED
    __shared__ __align__(1024) uint8_t sA[STEPS * A_STEP_BYTES];
    __shared__ __align__(1024) uint8_t sB[STEPS * B_STEP_BYTES];
    __shared__ __align__(8) uint64_t mbar;
    __shared__ uint32_t tmem_base_slot;

    const int tile = blockIdx.x;
    if (tile >= ntiles) return;
    const int tid = threadIdx.x, warp = tid >> 5;
    constexpr int K = K16 * STEPS;
    const uint16_t* At = A + (size_t)tile * M * K;
    const uint16_t* Bw = Bt + (size_t)tile * N * K;
    const uint32_t* Ct = C + (size_t)tile * M * N;

    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_base_slot)), "r"(TMEM_COLS) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    }
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&mbar)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    for (int s = 0; s < STEPS; s++) {
        store_row(sA + s * A_STEP_BYTES, tid, At + (size_t)tid * K + s * K16, lbo, sbo);
        if (tid < MMA_N) store_row(sB + s * B_STEP_BYTES, tid, Bw + (size_t)(tid % N) * K + s * K16, lbo, sbo);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t tmem_base = tmem_base_slot;
    const uint32_t lane_base = (uint32_t)(32 * (warp & 3)) << 16;  // a warp addresses its own 32-lane sub-partition

    {
        uint32_t r[MMA_N];
#pragma unroll
        for (int n = 0; n < MMA_N; n++) r[n] = Ct[tid * N + n % N];
        asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                     ::"r"(tmem_base + lane_base), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                     "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory");
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (tid == 0) {
        const uint32_t idesc = make_idesc();
#pragma unroll 1
        for (int s = 0; s < STEPS; s++) {
            const uint64_t da = make_smem_desc(smem_u32(sA + s * A_STEP_BYTES), lbo, sbo);
            const uint64_t db = make_smem_desc(smem_u32(sB + s * B_STEP_BYTES), lbo, sbo);
            asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                         "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(tmem_base), "l"(da), "l"(db), "r"(idesc),
                         "r"(s > 0 ? 1u : enable_first) : "memory");
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(&mbar)) : "memory");
    }

    {
        uint32_t done = 0;
        while (!done) {
            asm volatile("{\n.reg .pred P1;\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.b32 %0, 1, 0, P1;\n}"
                         : "=r"(done) : "r"(smem_u32(&mbar)), "r"(0u) : "memory");
        }
    }
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    {
        uint32_t r[MMA_N];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                       "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                     : "r"(tmem_base + lane_base) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t* out = D + ((size_t)tile * M + tid) * N;
        unsigned bad = 0;
#pragma unroll
        for (int n = 0; n < N; n++) {
            out[n] = r[n];
            bad += r[n] != r[n + N];
        }
        if (bad) atomicAdd(&g_replica_mismatches, (unsigned long long)bad);
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"(TMEM_COLS) : "memory");
    }
#else
    __trap();
#endif
}

__global__ void __launch_bounds__(THREADS) tcgen05_bf16_m128n8k16_tiles_kernel(const uint16_t* A, const uint16_t* Bt, const uint32_t* C,
                                                                               uint32_t* D, int ntiles, uint32_t lbo, uint32_t sbo,
                                                                               uint32_t enable_first) {
    tile_body<1>(A, Bt, C, D, ntiles, lbo, sbo, enable_first);
}

__global__ void __launch_bounds__(THREADS) tcgen05_bf16_m128n8k16_chain4_tiles_kernel(const uint16_t* A, const uint16_t* Bt, const uint32_t* C,
                                                                                      uint32_t* D, int ntiles, uint32_t lbo, uint32_t sbo,
                                                                                      uint32_t enable_first) {
    tile_body<MAX_STEPS>(A, Bt, C, D, ntiles, lbo, sbo, enable_first);
}

void launch_single(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    tcgen05_bf16_m128n8k16_tiles_kernel<<<ntiles, THREADS>>>(reinterpret_cast<const uint16_t*>(A), reinterpret_cast<const uint16_t*>(Bt), C, D,
                                                             ntiles, g_lbo, g_sbo, g_enable_input_d);
}

void launch_chain4(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    tcgen05_bf16_m128n8k16_chain4_tiles_kernel<<<ntiles, THREADS>>>(reinterpret_cast<const uint16_t*>(A), reinterpret_cast<const uint16_t*>(Bt), C, D,
                                                                    ntiles, g_lbo, g_sbo, g_enable_input_d);
}

}  // namespace

extern "C" {

// Sets the shared-memory descriptors' leading / stride dimension byte offsets for subsequent calls (defaults 128 / 256).
void tcgen05_set_descriptor_offsets(uint32_t lbo_bytes, uint32_t sbo_bytes) {
    g_lbo = lbo_bytes;
    g_sbo = sbo_bytes;
}

// The first MMA's enable_input_d for subsequent calls (default 1: D = A.B + C; 0: D = A.B, the accumulator not read).
void tcgen05_set_enable_input_d(uint32_t enable) { g_enable_input_d = enable ? 1u : 0u; }

// The words, since the last call, in which the instruction's columns 8..15 differed from its columns 0..7 (then reset to 0).
unsigned long long tcgen05_replica_mismatches(void) {
    unsigned long long v = 0, zero = 0;
    cudaMemcpyFromSymbol(&v, g_replica_mismatches, sizeof(v));
    cudaMemcpyToSymbol(g_replica_mismatches, &zero, sizeof(zero));
    return v;
}

// A: ntiles*128*16 BF16 words, Bt: ntiles*8*16 BF16 words, C/D: ntiles*128*8 f32 bits.  Returns the cudaError_t as an int.
int tcgen05_bf16_m128n8k16_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, M * K16 * 2, N * K16 * 2, M * N, launch_single);
}

// Four chained MMAs per tile on one TMEM accumulator: A ntiles*128*64 BF16 words, Bt ntiles*8*64, C/D ntiles*128*8 f32 bits.
int tcgen05_bf16_m128n8k16_chain4_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, M * K16 * 2 * MAX_STEPS, N * K16 * 2 * MAX_STEPS, M * N, launch_chain4);
}

}  // extern "C"
