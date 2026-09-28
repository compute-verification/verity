// Single-instruction tensor-core probes for Hopper `wgmma`: exactly one instruction per warpgroup per tile (sm_90a only).
//
//   wgmma_e4m3_m64n8k32_tiles   wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3   A 64x32 e4m3, Bt 8x32 e4m3, C/D 64x8 f32
//   wgmma_e5m2_m64n8k32_tiles   wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2   A 64x32 e5m2, Bt 8x32 e5m2, C/D 64x8 f32
//   wgmma_bf16_m64n8k16_tiles   wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16   A 64x16 bf16 (SS: A in shared memory)
//   wgmma_bf16_m64n8k16_rs_tiles                                                      the same with A from registers (RS)
//
// Written for lane tc-probe-fp8 of the Verity silicon campaign after veritor `gpu/tc-fp8/wgmma_fp8_probe.py`, which reached the
// instruction through Triton (`tl.dot` on a 64x64x32 tile) and confirmed a single QGMMA in the SASS; here the instruction is
// issued directly so that the probe needs only nvcc.  One warpgroup (128 threads, one block) executes exactly one
// `wgmma.mma_async` on one tile: every output word D[i][j] is the hardware's own function of (C[i][j], A[i][:], B[:][j]) with
// no software arithmetic in between.  Operands and results move as raw bits.
//
// Operand paths.  A comes from registers (the PTX ISA "wgmma .m64nNk32 register fragment layout for matrix A" with 8-bit types:
// warp w of the warpgroup holds rows 16w..16w+15 in the mma.m16n8k32 A layout, i.e. with g = lane>>2, t = lane&3 register i
// holds row 16w+g (+8 for odd i), k = 4t..4t+3 (+16 for i>=2)).  B comes from shared memory through a matrix descriptor in the
// K-major no-swizzle ("INTERLEAVE") canonical layout: core matrices of 8 rows x 16 bytes stored contiguously, the second
// core matrix along K at `lbo` bytes (leading dimension byte offset) and the next 8 rows -- absent here, N = 8 -- at `sbo`
// bytes (stride dimension byte offset); descriptor bits [0,14) = address>>4, [16,30) = lbo>>4, [32,46) = sbo>>4, [62,64) = 0
// (cute::GmmaDescriptor, make_gmma_desc<Major::K>: ((8,n),2):((1,SBO),LBO) in 16-byte units).  The driver verifies the layout
// with exact small-integer tiles before any semantic probe and may pass the swapped (lbo, sbo) if the first choice fails; the
// pair that passed is recorded.  The accumulator D is the PTX ISA m64nN f32 fragment: thread (w, lane) holds rows 16w+g and
// 16w+g+8, columns 2t and 2t+1 -- the mma.m16n8 C/D layout per warp.
//
// Build:  nvcc -O2 -arch=sm_90a -Xcompiler -fPIC -shared -o libwgmma_tiles_sm_90a.so wgmma_tiles.cu

#include "tiles_common.cuh"

namespace {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define TC_PROBE_WGMMA_ENABLED 1
#else
#define TC_PROBE_WGMMA_ENABLED 0
#endif

__device__ __forceinline__ uint64_t smem_desc(const void* smem_ptr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr & 0x3FFFF) >> 4);
    desc |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
    desc |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
    // base offset 0 (bits 49-51); layout type 0 = no swizzle (bits 62-63)
    return desc;
}

__device__ __forceinline__ void wgmma_fence() {
#if TC_PROBE_WGMMA_ENABLED
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_commit() {
#if TC_PROBE_WGMMA_ENABLED
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_wait_all() {
#if TC_PROBE_WGMMA_ENABLED
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void fence_operand(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }

// D += A * B with A in registers (4 x b32), B through the shared-memory descriptor, scale-d = 1 (accumulate into D).
__device__ __forceinline__ void wgmma_e4m3_m64n8k32(float d[4], const uint32_t a[4], uint64_t desc_b) {
#if TC_PROBE_WGMMA_ENABLED
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(1));
#else
    __trap();
#endif
}

__device__ __forceinline__ void wgmma_e5m2_m64n8k32(float d[4], const uint32_t a[4], uint64_t desc_b) {
#if TC_PROBE_WGMMA_ENABLED
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(1));
#else
    __trap();
#endif
}

// D += A * B, BF16, both operands through shared-memory descriptors (SS: the form cuBLAS / CUTLASS GEMM mainloops issue);
// scale-d = 1, imm-scale-a = imm-scale-b = 1, imm-trans-a = imm-trans-b = 0 (both K-major).
__device__ __forceinline__ void wgmma_bf16_m64n8k16_ss(float d[4], uint64_t desc_a, uint64_t desc_b) {
#if TC_PROBE_WGMMA_ENABLED
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(1));
#else
    __trap();
#endif
}

// D += A * B, BF16, A from registers (RS: the form FA3's P.V product issues), B through the descriptor; imm-trans-b = 0.
__device__ __forceinline__ void wgmma_bf16_m64n8k16_rs(float d[4], const uint32_t a[4], uint64_t desc_b) {
#if TC_PROBE_WGMMA_ENABLED
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(1));
#else
    __trap();
#endif
}

enum Op { E4M3 = 0, E5M2 = 1 };
constexpr int M = 64, N = 8, K = 32;
constexpr int THREADS = 128;  // one warpgroup

// One warpgroup (one block of 128 threads) per tile: A[64][32] bytes, Bt[8][32] bytes, C/D[64][8] u32.
template <int OP>
__device__ __forceinline__ void tile_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,
                                          const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int ntiles,
                                          uint32_t lbo, uint32_t sbo) {
    const int tile = blockIdx.x;
    if (tile >= ntiles) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    // B in the canonical K-major no-swizzle layout: byte (kh*128 + n*16 + kk) holds Bt[n][kh*16 + kk].
    __shared__ __align__(128) uint8_t sB[N * K];
    const uint8_t* a = A + (size_t)tile * M * K;
    const uint8_t* b = Bt + (size_t)tile * N * K;
    const uint32_t* c = C + (size_t)tile * M * N;
    uint32_t* d = D + (size_t)tile * M * N;
    for (int i = tid; i < N * K; i += THREADS) {
        const int kh = i >> 7, n = (i >> 4) & 7, kk = i & 15;
        sB[i] = b[n * K + kh * 16 + kk];
    }
#if TC_PROBE_WGMMA_ENABLED
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // generic-proxy writes -> visible to the async proxy
#endif
    __syncthreads();

    const int r0 = 16 * warp + g, r1 = r0 + 8;
    uint32_t af[4];
    af[0] = *reinterpret_cast<const uint32_t*>(a + r0 * K + 4 * t);
    af[1] = *reinterpret_cast<const uint32_t*>(a + r1 * K + 4 * t);
    af[2] = *reinterpret_cast<const uint32_t*>(a + r0 * K + 16 + 4 * t);
    af[3] = *reinterpret_cast<const uint32_t*>(a + r1 * K + 16 + 4 * t);
    float df[4];
    df[0] = __uint_as_float(c[r0 * N + 2 * t]);
    df[1] = __uint_as_float(c[r0 * N + 2 * t + 1]);
    df[2] = __uint_as_float(c[r1 * N + 2 * t]);
    df[3] = __uint_as_float(c[r1 * N + 2 * t + 1]);
    const uint64_t desc_b = smem_desc(sB, lbo, sbo);

    for (int i = 0; i < 4; ++i) fence_operand(df[i]);
    for (int i = 0; i < 4; ++i) fence_operand(af[i]);
    wgmma_fence();
    if (OP == E4M3) {
        wgmma_e4m3_m64n8k32(df, af, desc_b);
    } else {
        wgmma_e5m2_m64n8k32(df, af, desc_b);
    }
    wgmma_commit();
    wgmma_wait_all();
    for (int i = 0; i < 4; ++i) fence_operand(df[i]);

    d[r0 * N + 2 * t] = __float_as_uint(df[0]);
    d[r0 * N + 2 * t + 1] = __float_as_uint(df[1]);
    d[r1 * N + 2 * t] = __float_as_uint(df[2]);
    d[r1 * N + 2 * t + 1] = __float_as_uint(df[3]);
}

__global__ void wgmma_e4m3_m64n8k32_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles,
                                                 uint32_t lbo, uint32_t sbo) {
    tile_body<E4M3>(A, Bt, C, D, ntiles, lbo, sbo);
}
__global__ void wgmma_e5m2_m64n8k32_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles,
                                                 uint32_t lbo, uint32_t sbo) {
    tile_body<E5M2>(A, Bt, C, D, ntiles, lbo, sbo);
}

// BF16 m64n8k16: A[64][16*STEPS] and Bt[8][16*STEPS] BF16 words, C/D[64][8] u32; one warpgroup per tile.  STEPS = 1 is the
// single-instruction probe; STEPS = 4 issues four wgmma back to back on the same accumulator registers inside one
// fence / commit / wait group (k slices 0..3 in order), as a GEMM mainloop does for one K = 64 block.  Both shared-memory operands
// use the K-major no-swizzle layout of the FP8 B above, one slice after the other: core matrices of 8 rows x 16 bytes (8 BF16
// along K), the second core matrix along K at `lbo` bytes and the next 8-row group at `sbo` bytes.  A_SMEM selects SS (A through
// its own descriptor) or RS (A in the mma.m16n8k16 A-fragment registers: register i holds row 16w+g (+8 for odd i), k = 2t, 2t+1
// (+8 for i >= 2), the lower k in the low half).
constexpr int K16 = 16;

template <bool A_SMEM, int STEPS>
__device__ __forceinline__ void bf16_tile_body(const uint16_t* __restrict__ A, const uint16_t* __restrict__ Bt,
                                               const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int ntiles,
                                               uint32_t lbo, uint32_t sbo) {
    constexpr int KT = K16 * STEPS;
    const int tile = blockIdx.x;
    if (tile >= ntiles) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    // slice s, element e of sB (byte 2e = kh*128 + n*16 + 2kk) holds Bt[n][16s + kh*8 + kk]; of sA (byte 2e = rg*256 + kh*128 +
    // r*16 + 2kk) holds A[rg*8 + r][16s + kh*8 + kk]
    __shared__ __align__(128) uint16_t sB[STEPS][N * K16];
    __shared__ __align__(128) uint16_t sA[A_SMEM ? STEPS : 1][M * K16];
    const uint16_t* a = A + (size_t)tile * M * KT;
    const uint16_t* b = Bt + (size_t)tile * N * KT;
    const uint32_t* c = C + (size_t)tile * M * N;
    uint32_t* d = D + (size_t)tile * M * N;
    for (int s = 0; s < STEPS; ++s) {
        for (int e = tid; e < N * K16; e += THREADS) {
            const int kh = e >> 6, n = (e >> 3) & 7, kk = e & 7;
            sB[s][e] = b[n * KT + s * K16 + kh * 8 + kk];
        }
        if (A_SMEM) {
            for (int e = tid; e < M * K16; e += THREADS) {
                const int rg = e >> 7, kh = (e >> 6) & 1, r = (e >> 3) & 7, kk = e & 7;
                sA[s][e] = a[(rg * 8 + r) * KT + s * K16 + kh * 8 + kk];
            }
        }
    }
#if TC_PROBE_WGMMA_ENABLED
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
    __syncthreads();

    const int r0 = 16 * warp + g, r1 = r0 + 8;
    float df[4];
    df[0] = __uint_as_float(c[r0 * N + 2 * t]);
    df[1] = __uint_as_float(c[r0 * N + 2 * t + 1]);
    df[2] = __uint_as_float(c[r1 * N + 2 * t]);
    df[3] = __uint_as_float(c[r1 * N + 2 * t + 1]);
    uint32_t af[STEPS][4];
    if (!A_SMEM) {
        for (int s = 0; s < STEPS; ++s) {
            af[s][0] = *reinterpret_cast<const uint32_t*>(a + r0 * KT + s * K16 + 2 * t);
            af[s][1] = *reinterpret_cast<const uint32_t*>(a + r1 * KT + s * K16 + 2 * t);
            af[s][2] = *reinterpret_cast<const uint32_t*>(a + r0 * KT + s * K16 + 8 + 2 * t);
            af[s][3] = *reinterpret_cast<const uint32_t*>(a + r1 * KT + s * K16 + 8 + 2 * t);
            for (int i = 0; i < 4; ++i) fence_operand(af[s][i]);
        }
    }
    for (int i = 0; i < 4; ++i) fence_operand(df[i]);
    wgmma_fence();
#pragma unroll
    for (int s = 0; s < STEPS; ++s) {
        const uint64_t desc_b = smem_desc(sB[s], lbo, sbo);
        if (A_SMEM) {
            wgmma_bf16_m64n8k16_ss(df, smem_desc(sA[s], lbo, sbo), desc_b);
        } else {
            wgmma_bf16_m64n8k16_rs(df, af[s], desc_b);
        }
    }
    wgmma_commit();
    wgmma_wait_all();
    for (int i = 0; i < 4; ++i) fence_operand(df[i]);

    d[r0 * N + 2 * t] = __float_as_uint(df[0]);
    d[r0 * N + 2 * t + 1] = __float_as_uint(df[1]);
    d[r1 * N + 2 * t] = __float_as_uint(df[2]);
    d[r1 * N + 2 * t + 1] = __float_as_uint(df[3]);
}

#define BF16_KERNEL(NAME, A_SMEM, STEPS)                                                                                         \
    __global__ void NAME##_kernel(const uint16_t* A, const uint16_t* Bt, const uint32_t* C, uint32_t* D, int ntiles, uint32_t lbo, \
                                  uint32_t sbo) {                                                                                   \
        bf16_tile_body<A_SMEM, STEPS>(A, Bt, C, D, ntiles, lbo, sbo);                                                            \
    }
BF16_KERNEL(wgmma_bf16_m64n8k16_tiles, true, 1)
BF16_KERNEL(wgmma_bf16_m64n8k16_rs_tiles, false, 1)
BF16_KERNEL(wgmma_bf16_m64n8k16_chain4_tiles, true, 4)
BF16_KERNEL(wgmma_bf16_m64n8k16_rs_chain4_tiles, false, 4)

// The descriptor offsets of the current call (set by the extern "C" entry points before run_tiles launches).
uint32_t g_lbo = 128, g_sbo = 256;

#define BF16_LAUNCH(NAME)                                                                                                        \
    void launch_##NAME(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {                        \
        NAME##_kernel<<<ntiles, THREADS>>>(reinterpret_cast<const uint16_t*>(A), reinterpret_cast<const uint16_t*>(Bt), C, D, ntiles, \
                                           g_lbo, g_sbo);                                                                        \
    }
BF16_LAUNCH(wgmma_bf16_m64n8k16_tiles)
BF16_LAUNCH(wgmma_bf16_m64n8k16_rs_tiles)
BF16_LAUNCH(wgmma_bf16_m64n8k16_chain4_tiles)
BF16_LAUNCH(wgmma_bf16_m64n8k16_rs_chain4_tiles)

void launch_e4m3(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    wgmma_e4m3_m64n8k32_tiles_kernel<<<ntiles, THREADS>>>(A, Bt, C, D, ntiles, g_lbo, g_sbo);
}
void launch_e5m2(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    wgmma_e5m2_m64n8k32_tiles_kernel<<<ntiles, THREADS>>>(A, Bt, C, D, ntiles, g_lbo, g_sbo);
}

}  // namespace

extern "C" {

// Sets the B descriptor's leading / stride dimension byte offsets for subsequent calls (defaults 128 / 256).
void wgmma_set_descriptor_offsets(uint32_t lbo_bytes, uint32_t sbo_bytes) {
    g_lbo = lbo_bytes;
    g_sbo = sbo_bytes;
}

// A: ntiles*64*32 e4m3 bytes, Bt: ntiles*8*32 e4m3 bytes, C/D: ntiles*64*8 f32 bits.  Returns the cudaError_t as an int.
int wgmma_e4m3_m64n8k32_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, M * K, N * K, M * N, launch_e4m3);
}

// A: ntiles*64*32 e5m2 bytes, Bt: ntiles*8*32 e5m2 bytes, C/D: ntiles*64*8 f32 bits.
int wgmma_e5m2_m64n8k32_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, M * K, N * K, M * N, launch_e5m2);
}

// A: ntiles*64*16 BF16 words, Bt: ntiles*8*16 BF16 words, C/D: ntiles*64*8 f32 bits.  SS form (A through a descriptor).
int wgmma_bf16_m64n8k16_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, M * K16 * 2, N * K16 * 2, M * N, launch_wgmma_bf16_m64n8k16_tiles);
}

// The same tiles in the RS form (A from registers).
int wgmma_bf16_m64n8k16_rs_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, M * K16 * 2, N * K16 * 2, M * N, launch_wgmma_bf16_m64n8k16_rs_tiles);
}

// Four chained wgmma per tile: A ntiles*64*64 BF16 words, Bt ntiles*8*64, C/D ntiles*64*8 f32 bits (SS, then RS).
int wgmma_bf16_m64n8k16_chain4_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, M * K16 * 8, N * K16 * 8, M * N, launch_wgmma_bf16_m64n8k16_chain4_tiles);
}
int wgmma_bf16_m64n8k16_rs_chain4_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, M * K16 * 8, N * K16 * 8, M * N, launch_wgmma_bf16_m64n8k16_rs_chain4_tiles);
}

}  // extern "C"
