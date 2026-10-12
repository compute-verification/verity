// Blackwell datacenter (sm_100a / sm_103a) FP4 tensor-core probe: ONE `tcgen05.mma` per tile, raw bits in and out.
//
// Lane tcgen05.  The question is whether `tcgen05.mma.kind::mxf4nvf4` (tensor-memory accumulators, 5th-generation tensor
// cores) computes the same words as the consumer sm_120 `mma.sync.kind::mxf4nvf4` that lane fp4-re pinned with
// BlockScaledAlignAdd.  The harness (probe.py) keeps its 16x8xK=64 tiles: this kernel runs each tile as a 128x8x64
// `tcgen05.mma` (the smallest M the instruction supports for this kind) with the 16 tile rows REPLICATED into all eight
// 16-row blocks of A and of the TMEM accumulator, so the same 128 words come out of every replica -- rows 0..15 are the
// tile's D, rows 16..127 are a free consistency check across the four TMEM sub-partitions (each one a separate datapath).
//
// Per tile (one 128-thread CTA):
//   1. warp 0 allocates 32 TMEM columns (tcgen05.alloc); thread 0 initialises an mbarrier.
//   2. all threads write A (128 x 64 e2m1 nibbles) and B (8 x 64) into shared memory in the canonical K-major no-swizzle
//      layout the shared-memory descriptor describes: core matrices of 8 rows x 16 bytes, LBO between the two 16-byte K
//      chunks of a row group, SBO between 8-row groups (both runtime parameters so a misread of the descriptor spec can be
//      corrected from the host; the harness's exact-integer layout check catches any misrouting).
//   3. each warp preloads its 32 TMEM lanes: the accumulator (lane m, columns 0..7 = C[m % 16][0..7]) with tcgen05.st, and
//      the scale factors in the "32-lane" layout of the instruction descriptor (word column c of lane L holds the scale bytes
//      of row 32c + L%32, byte j = K-group j; replicated into all four sub-partitions; SFB likewise for column n, N padded
//      to 128 as CUTLASS does), one set per K step.
//   4. thread 0 issues the MMA(s) -- `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::{4X,2X}` with the
//      shared-memory descriptors, the instruction descriptor (E2M1 x E2M1 -> F32, scale UE4M3 / UE8M0, M=128, N=8,
//      K-major) and the per-tile `enable_input_d` predicate (0: D = A.B, the accumulator is not read; 1: D = A.B + D) --
//      then `tcgen05.commit` onto the mbarrier.
//   5. every thread waits on the mbarrier, its warp loads its 32 lanes x 8 columns of D with tcgen05.ld, writes them to
//      global memory (Dall[tile][128][8]); warp 0 frees the TMEM columns.
// With `steps` = 2 the tile is a K=128 chain: two MMAs back to back on the same TMEM accumulator (the second with
// enable_input_d = 1), i.e. the accumulation path a real GEMM main loop uses, without a host round trip.
//
// Host arrays (per tile, contiguous, little-endian): A codes u8[16][K] (low nibble), Bt u8[8][K] (B n-major, k contiguous),
// SFA u8[16][K/16 or K/32], SFB u8[8][...], C u32[16][8], enable u8, Dall u32[128][8].
// Build: nvcc -O2 -arch=sm_100a -Xcompiler -fPIC -shared -o libtcgen05_fp4_sm_100a.so tcgen05_fp4.cu
// (sm_103a for a B300; the arch-specific target is required: tcgen05 block-scaled kinds are `a`-feature instructions.)
//
// Descriptor bit layouts: PTX ISA "tcgen05 shared memory descriptor" (Table 49) and "instruction descriptor for
// .kind::mxf4 / .kind::mxf4nvf4" (Table 53), cross-checked against CUTLASS 3.9.2 cute/arch/mma_sm100_desc.hpp
// (UMMA::SmemDescriptor, UMMA::InstrDescriptorBlockScaled, make_umma_desc<Major::K> INTERLEAVE) and the TMEM scale-factor
// layout of cute/atom/mma_traits_sm100.hpp (tmem_sf_frg, ScaleFactorDuplicated4by1).

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int TILE_M = 16;  // harness tile rows
constexpr int TILE_N = 8;   // harness tile cols
constexpr int MMA_M = 128;  // tcgen05 M for kind::mxf4nvf4, cta_group::1 (the only valid M)
constexpr int MMA_N = 8;
constexpr int MMA_K = 64;
constexpr int MAX_STEPS = 2;
constexpr int A_BLOCK_BYTES = MMA_M * MMA_K / 2;  // 4096
constexpr int B_BLOCK_BYTES = MMA_N * MMA_K / 2;  // 256
constexpr int TMEM_COLS = 32;
constexpr int COL_D = 0;   // accumulator: columns 0..7
constexpr int COL_SF = 8;  // per step s: SFA at COL_SF + 8 s (4 words), SFB at COL_SF + 8 s + 4 (4 words)

struct SmemLayout {
    uint32_t lbo_a, sbo_a, lbo_b, sbo_b;  // bytes
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// PTX ISA Table 49: bits 0-13 start address >> 4, 16-29 leading dimension byte offset >> 4, 32-45 stride dimension byte
// offset >> 4, 46-48 fixed 0b001, 49-51 base offset (0: no swizzle), 52 lbo mode (0: relative), 61-63 swizzle (0: none).
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFFu) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFFu) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFFu) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    return d;
}

// PTX ISA Table 53 (.kind::mxf4 / .kind::mxf4nvf4): bit 2 sparsity 0, bits 7-9 atype E2M1 = 1, bits 10-11 btype E2M1 = 1,
// bits 15/16 transpose 0 (K-major), bits 17-22 N >> 3, bits 23-24 scale type (UE4M3 = 0, UE8M0 = 1), bits 27-28 M >> 7,
// bits 4-5 / 29-30 scale-factor data ids 0 (byte 0 of the TMEM word), bits 3/31 K = 64.
__host__ __device__ constexpr uint32_t make_idesc(bool scale_ue8m0) {
    return (1u << 7) | (1u << 10) | ((uint32_t)(MMA_N >> 3) << 17) | ((scale_ue8m0 ? 1u : 0u) << 23) | ((uint32_t)(MMA_M >> 7) << 27);
}

__device__ __forceinline__ void pack_row_chunk(uint32_t w[4], const uint8_t* codes) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
        uint32_t v = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) v |= (uint32_t)(codes[q * 8 + j] & 0xF) << (4 * j);  // element k -> nibble k % 8 of word k / 8
        w[q] = v;
    }
}

// KIND 0: nvf4 = kind::mxf4nvf4 .scale_vec::4X, UE4M3 (4 K-blocks of 16); KIND 1: mxf4 = kind::mxf4nvf4 .scale_vec::2X, UE8M0
// (2 blocks of 32); KIND 2: the same operands through kind::mxf4 .scale_vec::2X (CUTLASS's mnemonic for UE8M0 x E2M1).
template <int KIND>
__device__ __forceinline__ void tcgen05_tile_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,
                                                  const uint8_t* __restrict__ SFA, const uint8_t* __restrict__ SFB,
                                                  const uint32_t* __restrict__ C, const uint8_t* __restrict__ enable,
                                                  uint32_t* __restrict__ Dall, int ntiles, int steps, SmemLayout lay) {
    __shared__ __align__(1024) uint8_t sA[MAX_STEPS * A_BLOCK_BYTES];
    __shared__ __align__(1024) uint8_t sB[MAX_STEPS * B_BLOCK_BYTES];
    __shared__ __align__(8) uint64_t mbar;
    __shared__ uint32_t tmem_base_slot;

    const int tile = blockIdx.x;
    if (tile >= ntiles) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int NBLK = (KIND == 0) ? 4 : 2;
    const int K = MMA_K * steps, nblk_total = NBLK * steps;
    const uint8_t* At = A + (size_t)tile * TILE_M * K;
    const uint8_t* Bw = Bt + (size_t)tile * TILE_N * K;
    const uint8_t* sfa = SFA + (size_t)tile * TILE_M * nblk_total;
    const uint8_t* sfb = SFB + (size_t)tile * TILE_N * nblk_total;
    const uint32_t* Ct = C + (size_t)tile * TILE_M * TILE_N;
    const uint32_t use_acc = enable[tile] ? 1u : 0u;

    // 1. TMEM allocation (a full warp) and the completion mbarrier.
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_base_slot)), "r"(TMEM_COLS) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    }
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&mbar)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    // 2. Operand images: thread t owns row m = t of A (tile row m % 16); threads 0..7 own the B rows (columns n).
    for (int s = 0; s < steps; s++) {
        const int m = tid, i = m % TILE_M;
        const uint8_t* arow = At + (size_t)i * K + s * MMA_K;
        uint8_t* ablk = sA + s * A_BLOCK_BYTES;
#pragma unroll
        for (int c = 0; c < 2; c++) {
            uint32_t w[4];
            pack_row_chunk(w, arow + 32 * c);
            *reinterpret_cast<uint4*>(ablk + (m / 8) * lay.sbo_a + c * lay.lbo_a + (m % 8) * 16) = make_uint4(w[0], w[1], w[2], w[3]);
        }
        if (tid < TILE_N) {
            const int n = tid;
            const uint8_t* brow = Bw + (size_t)n * K + s * MMA_K;
            uint8_t* bblk = sB + s * B_BLOCK_BYTES;
#pragma unroll
            for (int c = 0; c < 2; c++) {
                uint32_t w[4];
                pack_row_chunk(w, brow + 32 * c);
                *reinterpret_cast<uint4*>(bblk + (n / 8) * lay.sbo_b + c * lay.lbo_b + (n % 8) * 16) = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic-proxy smem writes -> visible to the tensor core (async proxy)
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t tmem_base = tmem_base_slot;
    const uint32_t lane_base = (uint32_t)(32 * (warp & 3)) << 16;  // a warp addresses its own 32-lane sub-partition

    // 3. TMEM preload: accumulator lane m = tid (C[m % 16][0..7], or zero bits when the accumulator is disabled) and the
    //    scale factors per step.
    {
        const int i = tid % TILE_M;
        uint32_t r[8];
#pragma unroll
        for (int n = 0; n < 8; n++) r[n] = use_acc ? Ct[i * TILE_N + n] : 0u;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(tmem_base + lane_base + COL_D),
                     "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
        for (int s = 0; s < steps; s++) {
            uint32_t fa[4], fb[4];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int row = 32 * c + lane, ri = row % TILE_M;  // 32-lane layout: word column c of lane L <- row 32c + L%32
                uint32_t v = 0;
#pragma unroll
                for (int j = 0; j < NBLK; j++) v |= (uint32_t)sfa[ri * nblk_total + s * NBLK + j] << (8 * j);
                fa[c] = v;
                const int col = 32 * c + lane;
                uint32_t u = 0;
                if (col < TILE_N) {
#pragma unroll
                    for (int j = 0; j < NBLK; j++) u |= (uint32_t)sfb[col * nblk_total + s * NBLK + j] << (8 * j);
                }
                fb[c] = u;
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" ::"r"(tmem_base + lane_base + COL_SF + 8 * s),
                         "r"(fa[0]), "r"(fa[1]), "r"(fa[2]), "r"(fa[3]) : "memory");
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" ::"r"(tmem_base + lane_base + COL_SF + 8 * s + 4),
                         "r"(fb[0]), "r"(fb[1]), "r"(fb[2]), "r"(fb[3]) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    // 4. The MMA(s), issued by one thread, one commit.  `#pragma unroll 1` keeps exactly ONE tcgen05.mma instruction in the
    //    SASS of each kernel (stage 0 counts it), whichever `steps` is at runtime.
    if (tid == 0) {
#pragma unroll 1
        for (int s = 0; s < steps; s++) {
            const uint64_t da = make_smem_desc(smem_u32(sA + s * A_BLOCK_BYTES), lay.lbo_a, lay.sbo_a);
            const uint64_t db = make_smem_desc(smem_u32(sB + s * B_BLOCK_BYTES), lay.lbo_b, lay.sbo_b);
            const uint32_t idesc = make_idesc(KIND != 0);
            const uint32_t enable_d = (use_acc || s > 0) ? 1u : 0u;
            const uint32_t tsfa = tmem_base + COL_SF + 8 * s, tsfb = tsfa + 4, td = tmem_base + COL_D;
            if constexpr (KIND == 0) {
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p;\n}" ::"r"(td),
                    "l"(da), "l"(db), "r"(idesc), "r"(enable_d), "r"(tsfa), "r"(tsfb) : "memory");
            } else if constexpr (KIND == 1) {
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p;\n}" ::"r"(td),
                    "l"(da), "l"(db), "r"(idesc), "r"(enable_d), "r"(tsfa), "r"(tsfb) : "memory");
            } else {
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p;\n}" ::"r"(td),
                    "l"(da), "l"(db), "r"(idesc), "r"(enable_d), "r"(tsfa), "r"(tsfb) : "memory");
            }
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(&mbar)) : "memory");
    }

    // 5. Completion, then every warp reads its 32 lanes of D.
    {
        uint32_t done = 0;
        while (!done) {
            asm volatile("{\n.reg .pred P1;\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.b32 %0, 1, 0, P1;\n}"
                         : "=r"(done) : "r"(smem_u32(&mbar)), "r"(0u) : "memory");
        }
    }
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                     : "r"(tmem_base + lane_base + COL_D) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t* out = Dall + ((size_t)tile * MMA_M + tid) * MMA_N;
#pragma unroll
        for (int n = 0; n < 8; n++) out[n] = r[n];
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"(TMEM_COLS) : "memory");
    }
}

__global__ void __launch_bounds__(128) tcgen05_nvfp4_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB,
                                                                  const uint32_t* C, const uint8_t* enable, uint32_t* Dall, int ntiles, int steps,
                                                                  SmemLayout lay) {
    tcgen05_tile_body<0>(A, Bt, SFA, SFB, C, enable, Dall, ntiles, steps, lay);
}

__global__ void __launch_bounds__(128) tcgen05_mxfp4_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB,
                                                                  const uint32_t* C, const uint8_t* enable, uint32_t* Dall, int ntiles, int steps,
                                                                  SmemLayout lay) {
    tcgen05_tile_body<1>(A, Bt, SFA, SFB, C, enable, Dall, ntiles, steps, lay);
}

__global__ void __launch_bounds__(128) tcgen05_mxf4kind_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB,
                                                                     const uint32_t* C, const uint8_t* enable, uint32_t* Dall, int ntiles, int steps,
                                                                     SmemLayout lay) {
    tcgen05_tile_body<2>(A, Bt, SFA, SFB, C, enable, Dall, ntiles, steps, lay);
}

template <class T>
static cudaError_t upload(T** dptr, const void* src, size_t bytes) {
    cudaError_t err = cudaMalloc((void**)dptr, bytes ? bytes : 1);
    if (err != cudaSuccess) return err;
    if (bytes) err = cudaMemcpy(*dptr, src, bytes, cudaMemcpyHostToDevice);
    return err;
}

}  // namespace

extern "C" {

// kind: 0 nvf4 (SFA u8[n][16][4 steps], SFB u8[n][8][4 steps]), 1 mxf4 via kind::mxf4nvf4, 2 mxf4 via kind::mxf4 (SF ... [2 steps]).
// A u8[n][16][64 steps], Bt u8[n][8][64 steps], C u32[n][16][8], enable u8[n] (1: D = A.B + C through the TMEM accumulator,
// 0: D = A.B, the accumulator is not read), Dall u32[n][128][8] (all 128 TMEM lanes: rows 0..15 are the tile, the rest are
// replicas).  lbo/sbo: the shared-memory descriptor strides in bytes (canonical: LBO 128 = the second 16-byte K chunk of a
// row group, SBO 256 = the next 8-row group).  Returns the cudaError_t as an int.
int tcgen05_tiles(int kind, const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB, const uint32_t* C,
                  const uint8_t* enable, uint32_t* Dall, int n, int steps, unsigned lbo_a, unsigned sbo_a, unsigned lbo_b, unsigned sbo_b) {
    if (steps < 1 || steps > MAX_STEPS || kind < 0 || kind > 2 || n <= 0) return (int)cudaErrorInvalidValue;
    const int nblk = (kind == 0 ? 4 : 2) * steps;
    const size_t K = (size_t)MMA_K * steps;
    uint8_t *dA = nullptr, *dB = nullptr, *dSFA = nullptr, *dSFB = nullptr, *dEn = nullptr;
    uint32_t *dC = nullptr, *dD = nullptr;
    const size_t d_bytes = (size_t)n * MMA_M * MMA_N * 4;
    cudaError_t err = cudaSuccess;
    if ((err = upload(&dA, A, (size_t)n * TILE_M * K)) != cudaSuccess) return (int)err;
    if ((err = upload(&dB, Bt, (size_t)n * TILE_N * K)) != cudaSuccess) return (int)err;
    if ((err = upload(&dSFA, SFA, (size_t)n * TILE_M * nblk)) != cudaSuccess) return (int)err;
    if ((err = upload(&dSFB, SFB, (size_t)n * TILE_N * nblk)) != cudaSuccess) return (int)err;
    if ((err = upload(&dC, C, (size_t)n * TILE_M * TILE_N * 4)) != cudaSuccess) return (int)err;
    if ((err = upload(&dEn, enable, (size_t)n)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc((void**)&dD, d_bytes)) != cudaSuccess) return (int)err;
    cudaMemset(dD, 0xEE, d_bytes);  // a lane the MMA never wrote shows up as 0xEEEEEEEE, not as a plausible zero
    const SmemLayout lay{lbo_a, sbo_a, lbo_b, sbo_b};
    switch (kind) {
        case 0: tcgen05_nvfp4_tiles_kernel<<<n, 128>>>(dA, dB, dSFA, dSFB, dC, dEn, dD, n, steps, lay); break;
        case 1: tcgen05_mxfp4_tiles_kernel<<<n, 128>>>(dA, dB, dSFA, dSFB, dC, dEn, dD, n, steps, lay); break;
        default: tcgen05_mxf4kind_tiles_kernel<<<n, 128>>>(dA, dB, dSFA, dSFB, dC, dEn, dD, n, steps, lay); break;
    }
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(Dall, dD, d_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dSFA);
    cudaFree(dSFB);
    cudaFree(dC);
    cudaFree(dEn);
    cudaFree(dD);
    return (int)err;
}

// The instruction descriptors the kernels use (for the record): kind 0 -> UE4M3, otherwise UE8M0.
unsigned tcgen05_idesc(int kind) { return make_idesc(kind != 0); }

int has_mxf4(void) { return 1; }

// The device's compute capability and name; for the capture's provenance.
int device_identity(int* major, int* minor, char* name, int name_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    *major = prop.major;
    *minor = prop.minor;
    int i = 0;
    for (; i < name_len - 1 && prop.name[i]; ++i) name[i] = prop.name[i];
    name[i] = 0;
    return 0;
}

}  // extern "C"
