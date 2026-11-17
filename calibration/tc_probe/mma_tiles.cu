// Single-instruction tensor-core probes for `mma.sync`: exactly one instruction per warp per tile.
//
//   mma_bf16_m16n8k16_tiles   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32   (sm_80+)   A 16x16 bf16, Bt 8x16 bf16
//   mma_e4m3_m16n8k32_tiles   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32   (sm_89+)   A 16x32 e4m3, Bt 8x32 e4m3
//   mma_e5m2_m16n8k32_tiles   mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32   (sm_89+)   A 16x32 e5m2, Bt 8x32 e5m2
//
// Provenance: veritor `gpu/tensor-core-semantics/mma_tiles.cu` (BF16 and E4M3 halves; the E5M2 kernel is the E4M3 one with the
// other operand type).  Every warp
// executes exactly one `mma.sync` on one tile, so each output element D[i][j] is the hardware's own function of
// (C[i][j], A[i][:], B[:][j]) with no software reduction in between.  Operands and results move as raw bits; nothing is
// converted or rounded on the way in or out.
//
//   A: 16xK row-major operand words,  Bt: 8xK (B stored n-major, k contiguous, i.e. the ".col" B operand),  C/D: 16x8 f32.
//
// Fragment layouts follow the PTX ISA "Matrix Fragments for mma.m16n8k16" (floating point) and "mma.m16n8k32" (8-bit) figures:
// with g = lane>>2, t = lane&3 and E = elements per 32-bit register (2 for BF16, 4 for FP8), register i of A holds row g (+8
// for odd i), columns t*E .. t*E+E-1 (+K/2 for i>=2); register i of B holds column g, the same k range.  The layout is verified
// by the driver (exact small-integer tiles) before any semantic probe runs.
//
// Build (sm_80 for the A100, sm_89 for the RTX 4090 / L40S, sm_90a for the H100):
//   nvcc -O2 -arch=sm_89 -Xcompiler -fPIC -shared -o libmma_tiles_sm_89.so mma_tiles.cu
// The FP8 kernels need sm_89+; for lower targets their bodies compile to a trap so that one translation unit serves every arch.

#include "tiles_common.cuh"

namespace {

__device__ __forceinline__ void mma_bf16_m16n8k16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void mma_e4m3_m16n8k32(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    __trap();
#endif
}

__device__ __forceinline__ void mma_e5m2_m16n8k32(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    __trap();
#endif
}

enum Op { BF16_K16 = 0, E4M3_K32 = 1, E5M2_K32 = 2 };

// One warp per tile.  A tile is A[16][K], Bt[8][K] (operand words of BYTES bytes), C[16][8] (u32 bits).
template <int K, int BYTES, int OP>
__device__ __forceinline__ void tile_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,
                                          const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int ntiles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    constexpr int E = 4 / BYTES;    // elements per 32-bit register
    constexpr int ROW = K * BYTES;  // bytes per row of A / Bt

    const uint8_t* a = A + (size_t)warp * 16 * ROW;
    const uint8_t* b = Bt + (size_t)warp * 8 * ROW;
    const uint32_t* c = C + (size_t)warp * 128;
    uint32_t* d = D + (size_t)warp * 128;

    uint32_t af[4], bf[2];
    af[0] = *reinterpret_cast<const uint32_t*>(a + (g) * ROW + (t * E) * BYTES);
    af[1] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * ROW + (t * E) * BYTES);
    af[2] = *reinterpret_cast<const uint32_t*>(a + (g) * ROW + (t * E + K / 2) * BYTES);
    af[3] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * ROW + (t * E + K / 2) * BYTES);
    bf[0] = *reinterpret_cast<const uint32_t*>(b + g * ROW + (t * E) * BYTES);
    bf[1] = *reinterpret_cast<const uint32_t*>(b + g * ROW + (t * E + K / 2) * BYTES);

    float cf[4], df[4];
    cf[0] = __uint_as_float(c[g * 8 + 2 * t]);
    cf[1] = __uint_as_float(c[g * 8 + 2 * t + 1]);
    cf[2] = __uint_as_float(c[(g + 8) * 8 + 2 * t]);
    cf[3] = __uint_as_float(c[(g + 8) * 8 + 2 * t + 1]);

    if (OP == BF16_K16) {
        mma_bf16_m16n8k16(df, af, bf, cf);
    } else if (OP == E4M3_K32) {
        mma_e4m3_m16n8k32(df, af, bf, cf);
    } else {
        mma_e5m2_m16n8k32(df, af, bf, cf);
    }

    d[g * 8 + 2 * t] = __float_as_uint(df[0]);
    d[g * 8 + 2 * t + 1] = __float_as_uint(df[1]);
    d[(g + 8) * 8 + 2 * t] = __float_as_uint(df[2]);
    d[(g + 8) * 8 + 2 * t + 1] = __float_as_uint(df[3]);
}

// One named kernel per instruction so that the SASS of each can be inspected on its own (cuobjdump -sass: one MMA each).
__global__ void mma_bf16_m16n8k16_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    tile_body<16, 2, BF16_K16>(A, Bt, C, D, ntiles);
}
__global__ void mma_e4m3_m16n8k32_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    tile_body<32, 1, E4M3_K32>(A, Bt, C, D, ntiles);
}
__global__ void mma_e5m2_m16n8k32_tiles_kernel(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    tile_body<32, 1, E5M2_K32>(A, Bt, C, D, ntiles);
}

constexpr int WARPS_PER_BLOCK = 4;

void launch_bf16(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    mma_bf16_m16n8k16_tiles_kernel<<<(ntiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 32 * WARPS_PER_BLOCK>>>(A, Bt, C, D, ntiles);
}
void launch_e4m3(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    mma_e4m3_m16n8k32_tiles_kernel<<<(ntiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 32 * WARPS_PER_BLOCK>>>(A, Bt, C, D, ntiles);
}
void launch_e5m2(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    mma_e5m2_m16n8k32_tiles_kernel<<<(ntiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 32 * WARPS_PER_BLOCK>>>(A, Bt, C, D, ntiles);
}

}  // namespace

extern "C" {

// A: ntiles*16*16 bf16 bits, Bt: ntiles*8*16 bf16 bits, C/D: ntiles*16*8 f32 bits.  Returns the cudaError_t as an int.
int mma_bf16_m16n8k16_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, 16 * 16 * 2, 8 * 16 * 2, 128, launch_bf16);
}

// A: ntiles*16*32 e4m3 bytes, Bt: ntiles*8*32 e4m3 bytes, C/D: ntiles*16*8 f32 bits.
int mma_e4m3_m16n8k32_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, 16 * 32, 8 * 32, 128, launch_e4m3);
}

// A: ntiles*16*32 e5m2 bytes, Bt: ntiles*8*32 e5m2 bytes, C/D: ntiles*16*8 f32 bits.
int mma_e5m2_m16n8k32_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    return tc_probe::run_tiles(A, Bt, C, D, ntiles, 16 * 32, 8 * 32, 128, launch_e5m2);
}

}  // extern "C"
