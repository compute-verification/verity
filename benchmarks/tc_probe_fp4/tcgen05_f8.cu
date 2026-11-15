// Blackwell datacenter (sm_100a / sm_103a) FP8 tensor-core probe: S chained `tcgen05.mma` per tile on one TMEM accumulator, raw
// bits in and out.
//
// The question is the bit-exact step of `tcgen05.mma.cta_group::1.kind::f8f6f4` (E4M3 / E5M2 operands, F32
// accumulator, K = 32 an instruction) and of `kind::mxf8f6f4.block_scale.scale_vec::1X` (the same with one UE8M0 scale a row of A
// and a column of B for each K = 32 block); the capture found sm_120's QMMA step (silicon.models
// BLACKWELL_SM100_E4M3 / _E5M2, f8_model.py for the mixed and scaled kinds).  It is tcgen05_fp4.cu's kernel with 8-bit
// elements, distinct rows (no replicas) and N = 32:
//
// Per tile (one 128-thread CTA), M = 128 rows, N = 32 columns, K = 32 * steps:
//   1. warp 0 allocates 128 TMEM columns; thread 0 initialises an mbarrier.
//   2. every thread writes its row of A (and threads 0..31 their row of B, B n-major) for each step into shared memory in the
//      canonical K-major no-swizzle layout: core matrices of 8 rows x 16 bytes, LBO 128 between the two 16-byte K chunks of a row
//      group, SBO 256 between 8-row groups (tcgen05_fp4.cu's layout, validated there: a K = 32 row of 8-bit codes is
//      32 bytes, as a K = 64 row of nibbles is).
//   3. each warp preloads its 32 TMEM lanes: the accumulator (lane m, columns 0..31 = C[m][0..31]) with tcgen05.st, and for
//      mxf8f6f4 the scale factors in tcgen05_fp4.cu's 32-lane layout (word column c of lane L holds the scale byte of row 32c + L,
//      byte 0; replicated into all four sub-partitions; SFB likewise for column n, N padded to 128), one set per step.
//   4. thread 0 issues the `steps` MMAs (the first with the tile's enable_input_d: 0 means D = A.B and the accumulator is not
//      read; the rest with 1), then one tcgen05.commit onto the mbarrier.  `#pragma unroll 1` keeps one MMA in the SASS.
//   5. every thread waits on the mbarrier; warp w loads lanes 32w..32w+31 x 32 columns of D with tcgen05.ld and writes them out.
//
// Host arrays (per tile, contiguous): A u8[128][32 steps], Bt u8[32][32 steps], SFA u8[128][steps], SFB u8[32][steps],
// C u32[128][32], enable u8, D u32[128][32].  The instruction descriptor is a runtime argument (the type pair lives in it).
// Build: nvcc -O2 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared -o libtcgen05_f8.so tcgen05_f8.cu
//
// Instruction descriptors, PTX ISA 9.7.18.4 (cross-checked with benchmarks/pouw/dc/dc.cuh, whose kind::f8f6f4 E4M3 descriptor
// LOWBIT-DC's fpx ran, and with tcgen05_fp4.cu's block-scaled one):
//   kind::f8f6f4:   D format [4,6) (F32 = 1), A format [7,10), B format [10,13) (E4M3 0, E5M2 1, E2M3 3, E3M2 4, E2M1 5),
//                   K-major, N >> 3 at [17,23), M >> 4 at [24,29).
//   kind::mxf8f6f4: A/B format as above at [7,10)/[10,13), N >> 3 at [17,23), scale type [23] (UE8M0 = 1), M >> 7 at [27,29),
//                   scale-factor data ids 0.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int MMA_M = 128;
constexpr int MMA_N = 32;
constexpr int MMA_K = 32;
constexpr int MAX_STEPS = 8;
constexpr int A_BLOCK_BYTES = MMA_M * MMA_K;  // 4096
constexpr int B_BLOCK_BYTES = MMA_N * MMA_K;  // 1024
constexpr int TMEM_COLS = 128;
constexpr int COL_D = 0;    // accumulator: columns 0..31
constexpr int COL_SF = 32;  // per step s: SFA at COL_SF + 8 s (4 words), SFB at COL_SF + 8 s + 4 (4 words)
constexpr uint32_t LBO = 128, SBO = 256;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// PTX ISA "tcgen05 shared memory descriptor": start >> 4 [0,14), LBO >> 4 [16,30), SBO >> 4 [32,46), fixed 0b001 at [46,49),
// base offset 0, relative LBO, no swizzle.
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t addr) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFFu) >> 4);
    d |= (uint64_t)((LBO & 0x3FFFFu) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFFu) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    return d;
}

__host__ __device__ constexpr uint32_t idesc_f8f6f4(uint32_t afmt, uint32_t bfmt) {
    return (1u << 4) | (afmt << 7) | (bfmt << 10) | ((uint32_t)(MMA_N >> 3) << 17) | ((uint32_t)(MMA_M >> 4) << 24);
}

__host__ __device__ constexpr uint32_t idesc_mxf8f6f4(uint32_t afmt, uint32_t bfmt) {
    return (afmt << 7) | (bfmt << 10) | ((uint32_t)(MMA_N >> 3) << 17) | (1u << 23) | ((uint32_t)(MMA_M >> 7) << 27);
}

// One 32-byte K row (two 16-byte chunks) of 8-bit codes from global memory (byte loads: no alignment assumed) into the
// core-matrix layout; element k is byte k % 16 of chunk k / 16.
__device__ __forceinline__ void put_row(uint8_t* blk, int r, const uint8_t* codes) {
    uint32_t w[8];
#pragma unroll
    for (int q = 0; q < 8; q++)
        w[q] = (uint32_t)codes[4 * q] | ((uint32_t)codes[4 * q + 1] << 8) | ((uint32_t)codes[4 * q + 2] << 16) | ((uint32_t)codes[4 * q + 3] << 24);
#pragma unroll
    for (int c = 0; c < 2; c++)
        *reinterpret_cast<uint4*>(blk + (r / 8) * SBO + c * LBO + (r % 8) * 16) = make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]);
}

// KIND 0: kind::f8f6f4; KIND 1: kind::mxf8f6f4.block_scale.scale_vec::1X.
template <int KIND>
__device__ __forceinline__ void tile_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt, const uint8_t* __restrict__ SFA,
                                          const uint8_t* __restrict__ SFB, const uint32_t* __restrict__ C,
                                          const uint8_t* __restrict__ enable, uint32_t* __restrict__ D, int ntiles, int steps,
                                          uint32_t idesc) {
    __shared__ __align__(1024) uint8_t sA[MAX_STEPS * A_BLOCK_BYTES];
    __shared__ __align__(1024) uint8_t sB[MAX_STEPS * B_BLOCK_BYTES];
    __shared__ __align__(8) uint64_t mbar;
    __shared__ uint32_t tmem_base_slot;

    const int tile = blockIdx.x;
    if (tile >= ntiles) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int K = MMA_K * steps;
    const uint8_t* At = A + (size_t)tile * MMA_M * K;
    const uint8_t* Bw = Bt + (size_t)tile * MMA_N * K;
    const uint8_t* sfa = SFA + (size_t)tile * MMA_M * steps;
    const uint8_t* sfb = SFB + (size_t)tile * MMA_N * steps;
    const uint32_t* Ct = C + (size_t)tile * MMA_M * MMA_N;
    const uint32_t use_acc = enable[tile] ? 1u : 0u;

    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_base_slot)), "r"(TMEM_COLS) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    }
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&mbar)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    // Operand images.  Global rows are K = 32 steps long; each step's 32 bytes go to that step's block.
    for (int s = 0; s < steps; s++) {
        put_row(sA + s * A_BLOCK_BYTES, tid, At + (size_t)tid * K + s * MMA_K);
        if (tid < MMA_N) put_row(sB + s * B_BLOCK_BYTES, tid, Bw + (size_t)tid * K + s * MMA_K);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t tmem_base = tmem_base_slot;
    const uint32_t lane_base = (uint32_t)(32 * (warp & 3)) << 16;

    {
        const int m = tid;
        uint32_t r[32];
#pragma unroll
        for (int n = 0; n < 32; n++) r[n] = use_acc ? Ct[m * MMA_N + n] : 0u;
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,"
            "%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(tmem_base + lane_base + COL_D),
            "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]),
            "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]),
            "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]),
            "r"(r[31])
            : "memory");
        if constexpr (KIND == 1) {
            for (int s = 0; s < steps; s++) {
                uint32_t fa[4], fb[4];
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int row = 32 * c + lane;
                    fa[c] = (uint32_t)sfa[row * steps + s];
                    fb[c] = row < MMA_N ? (uint32_t)sfb[row * steps + s] : 0u;
                }
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" ::"r"(tmem_base + lane_base + COL_SF + 8 * s),
                             "r"(fa[0]), "r"(fa[1]), "r"(fa[2]), "r"(fa[3]) : "memory");
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" ::"r"(tmem_base + lane_base + COL_SF + 8 * s + 4),
                             "r"(fb[0]), "r"(fb[1]), "r"(fb[2]), "r"(fb[3]) : "memory");
            }
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (tid == 0) {
#pragma unroll 1
        for (int s = 0; s < steps; s++) {
            const uint64_t da = make_smem_desc(smem_u32(sA + s * A_BLOCK_BYTES));
            const uint64_t db = make_smem_desc(smem_u32(sB + s * B_BLOCK_BYTES));
            const uint32_t enable_d = (use_acc || s > 0) ? 1u : 0u;
            const uint32_t td = tmem_base + COL_D;
            if constexpr (KIND == 0) {
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}" ::"r"(td),
                    "l"(da), "l"(db), "r"(idesc), "r"(enable_d) : "memory");
            } else {
                const uint32_t tsfa = tmem_base + COL_SF + 8 * s, tsfb = tsfa + 4;
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%5], [%6], p;\n}" ::"r"(td),
                    "l"(da), "l"(db), "r"(idesc), "r"(enable_d), "r"(tsfa), "r"(tsfb) : "memory");
            }
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
        uint32_t r[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,"
            "%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
              "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]),
              "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
              "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
            : "r"(tmem_base + lane_base + COL_D)
            : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t* out = D + ((size_t)tile * MMA_M + tid) * MMA_N;
#pragma unroll
        for (int n = 0; n < 32; n++) out[n] = r[n];
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"(TMEM_COLS) : "memory");
    }
}

__global__ void __launch_bounds__(128) tcgen05_f8f6f4_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB,
                                                                   const uint32_t* C, const uint8_t* enable, uint32_t* D, int ntiles, int steps,
                                                                   uint32_t idesc) {
    tile_body<0>(A, Bt, SFA, SFB, C, enable, D, ntiles, steps, idesc);
}

__global__ void __launch_bounds__(128) tcgen05_mxf8f6f4_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB,
                                                                     const uint32_t* C, const uint8_t* enable, uint32_t* D, int ntiles, int steps,
                                                                     uint32_t idesc) {
    tile_body<1>(A, Bt, SFA, SFB, C, enable, D, ntiles, steps, idesc);
}

struct Buffers {
    uint8_t *A = nullptr, *B = nullptr, *SFA = nullptr, *SFB = nullptr, *en = nullptr;
    uint32_t *C = nullptr, *D = nullptr;
    size_t cap_tiles = 0;
    int cap_steps = 0;
};
Buffers g_buf;

void release() {
    cudaFree(g_buf.A);
    cudaFree(g_buf.B);
    cudaFree(g_buf.SFA);
    cudaFree(g_buf.SFB);
    cudaFree(g_buf.en);
    cudaFree(g_buf.C);
    cudaFree(g_buf.D);
    g_buf = Buffers{};
}

cudaError_t reserve(size_t n, int steps) {
    if (n <= g_buf.cap_tiles && steps <= g_buf.cap_steps) return cudaSuccess;
    release();
    const size_t K = (size_t)MMA_K * steps;
    cudaError_t err;
    if ((err = cudaMalloc((void**)&g_buf.A, n * MMA_M * K)) != cudaSuccess) return err;
    if ((err = cudaMalloc((void**)&g_buf.B, n * MMA_N * K)) != cudaSuccess) return err;
    if ((err = cudaMalloc((void**)&g_buf.SFA, n * MMA_M * steps)) != cudaSuccess) return err;
    if ((err = cudaMalloc((void**)&g_buf.SFB, n * MMA_N * steps)) != cudaSuccess) return err;
    if ((err = cudaMalloc((void**)&g_buf.en, n)) != cudaSuccess) return err;
    if ((err = cudaMalloc((void**)&g_buf.C, n * MMA_M * MMA_N * 4)) != cudaSuccess) return err;
    if ((err = cudaMalloc((void**)&g_buf.D, n * MMA_M * MMA_N * 4)) != cudaSuccess) return err;
    g_buf.cap_tiles = n;
    g_buf.cap_steps = steps;
    return cudaSuccess;
}

}  // namespace

extern "C" {

// kind 0: kind::f8f6f4, kind 1: kind::mxf8f6f4.block_scale.scale_vec::1X.  A u8[n][128][32 steps], Bt u8[n][32][32 steps],
// SFA u8[n][128][steps], SFB u8[n][32][steps] (read for kind 1 only), C u32[n][128][32], enable u8[n], D u32[n][128][32].
// D is filled with 0xEEEEEEEE before the launch, so a word the MMA never wrote cannot pass as a plausible value.
int tcgen05_f8_tiles(int kind, unsigned idesc, const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB, const uint32_t* C,
                     const uint8_t* enable, uint32_t* D, int n, int steps) {
    if (steps < 1 || steps > MAX_STEPS || kind < 0 || kind > 1 || n <= 0) return (int)cudaErrorInvalidValue;
    const size_t K = (size_t)MMA_K * steps, nn = (size_t)n;
    cudaError_t err = reserve(nn, steps);
    if (err != cudaSuccess) return (int)err;
    const size_t d_bytes = nn * MMA_M * MMA_N * 4;
    if ((err = cudaMemcpy(g_buf.A, A, nn * MMA_M * K, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)err;
    if ((err = cudaMemcpy(g_buf.B, Bt, nn * MMA_N * K, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)err;
    if ((err = cudaMemcpy(g_buf.SFA, SFA, nn * MMA_M * steps, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)err;
    if ((err = cudaMemcpy(g_buf.SFB, SFB, nn * MMA_N * steps, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)err;
    if ((err = cudaMemcpy(g_buf.C, C, d_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)err;
    if ((err = cudaMemcpy(g_buf.en, enable, nn, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)err;
    if ((err = cudaMemset(g_buf.D, 0xEE, d_bytes)) != cudaSuccess) return (int)err;
    if (kind == 0)
        tcgen05_f8f6f4_tiles_kernel<<<n, 128>>>(g_buf.A, g_buf.B, g_buf.SFA, g_buf.SFB, g_buf.C, g_buf.en, g_buf.D, n, steps, idesc);
    else
        tcgen05_mxf8f6f4_tiles_kernel<<<n, 128>>>(g_buf.A, g_buf.B, g_buf.SFA, g_buf.SFB, g_buf.C, g_buf.en, g_buf.D, n, steps, idesc);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(D, g_buf.D, d_bytes, cudaMemcpyDeviceToHost);
    return (int)err;
}

void tcgen05_f8_release(void) { release(); }

unsigned tcgen05_f8_idesc(int kind, unsigned afmt, unsigned bfmt) { return kind == 0 ? idesc_f8f6f4(afmt, bfmt) : idesc_mxf8f6f4(afmt, bfmt); }

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
