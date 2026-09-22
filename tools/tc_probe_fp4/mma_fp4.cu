// Single-instruction Blackwell (sm_120) tensor-core probes: one `mma.sync` per warp per tile, raw bits in and out.
//
// Lane fp4-5090.  Every kernel here executes EXACTLY ONE mma.sync on one tile, so each output element D[i][j] is the
// hardware's own function of (C[i][j], A[i][:], B[:][j], the scales of row i / column j).  Operands and results move as
// raw bits; nothing is converted or rounded on the way in or out.  Chains (K > one instruction) are driven from the host
// by feeding D back as C, so the one-MMA-per-tile property is kept and checked by `--sass-check` (cuobjdump).
//
// Instructions (mnemonics copied verbatim from autoproof's pinned probe kernel probes/cuda/probe_mma.cu, whose SASS hash is in
// spec/SEMANTICS.lock.v1.json; the ue8m0 variant is ours):
//   nvfp4 : mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//   mxfp4 : mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0  (-DNO_MXF4 drops it)
//   fp4raw: mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32
//   bf16  : mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Fragment layouts (PTX ISA, cross-checked by autoproof against CUTLASS mma_traits_sm120.hpp; verified again by the driver's
// exact small-integer layout check before any semantic probe runs):
//   e2m1 A (16xK nibbles, 4 regs): reg r, nibble j -> row = lane/4 + 8*(r&1), k = 8*(lane%4) + j + 32*(r>>1)
//   e2m1 B (Kx8, 2 regs):          reg r, nibble j -> col = lane/4,            k = 8*(lane%4) + j + 32*r
//   SFA (.b32, byte b = K-block b): lane supplies row 8*(lane&1)+(lane>>2); lane bit 1 broadcast (thread-id selector 0)
//   SFB:                            lane supplies col lane>>2; lane bits 0-1 broadcast
//   f8f6f4 e2m1 (8-bit containers, nibble at bits [2,6)): A reg r byte j -> row lane/4 + 8*(r&1), k = 4*(lane%4)+j+16*(r>>1);
//                                                        B reg r byte j -> col lane/4, k = 4*(lane%4)+j+16*r
//   D/C (4 f32): d0/d1 -> row lane/4, cols 2*(lane%4)+{0,1}; d2/d3 -> row+8
//
// Host arrays (per tile, contiguous, little-endian): A codes u8[16][K] (low nibble), Bt codes u8[8][K] (B n-major, k contiguous),
// SFA u8[16][K/blk], SFB u8[8][K/blk], C/D u32[16][8].  Build: nvcc -O2 -arch=sm_120 -Xcompiler -fPIC -shared -o libmma_fp4.so mma_fp4.cu

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ void mma_nvfp4_m16n8k64(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4],
                                                   uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {0, 0}, {%15}, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa), "r"(sfb));
}

#ifndef NO_MXF4
__device__ __forceinline__ void mma_mxfp4_m16n8k64(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4],
                                                   uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {0, 0}, {%15}, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa), "r"(sfb));
}
#endif

__device__ __forceinline__ void mma_fp4raw_m16n8k32(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void mma_bf16_m16n8k16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ---- e2m1 nibble fragments (K = 64) ----
__device__ __forceinline__ void pack_a_e2m1(uint32_t a[4], const uint8_t* A, int lane) {
    const int row0 = lane >> 2, kcol = 8 * (lane & 3);
#pragma unroll
    for (int r = 0; r < 4; r++) {
        const int row = row0 + 8 * (r & 1), k0 = kcol + 32 * (r >> 1);
        uint32_t v = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) v |= (uint32_t)(A[row * 64 + k0 + j] & 0xF) << (4 * j);
        a[r] = v;
    }
}

__device__ __forceinline__ void pack_b_e2m1(uint32_t b[2], const uint8_t* Bt, int lane) {
    const int col = lane >> 2, kcol = 8 * (lane & 3);
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int k0 = kcol + 32 * r;
        uint32_t v = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) v |= (uint32_t)(Bt[col * 64 + k0 + j] & 0xF) << (4 * j);
        b[r] = v;
    }
}

// Scale words: byte b = K-block b of this lane's row / column (nblk = 4 for ue4m3 .4X, 2 for ue8m0 .2X; unused bytes zero).
__device__ __forceinline__ uint32_t pack_sfa(const uint8_t* SFA, int nblk, int lane) {
    const int row = 8 * (lane & 1) + (lane >> 2);
    uint32_t v = 0;
    for (int b = 0; b < nblk; b++) v |= (uint32_t)SFA[row * nblk + b] << (8 * b);
    return v;
}

__device__ __forceinline__ uint32_t pack_sfb(const uint8_t* SFB, int nblk, int lane) {
    const int col = lane >> 2;
    uint32_t v = 0;
    for (int b = 0; b < nblk; b++) v |= (uint32_t)SFB[col * nblk + b] << (8 * b);
    return v;
}

__device__ __forceinline__ void load_c(float cf[4], const uint32_t* c, int lane) {
    const int g = lane >> 2, t = lane & 3;
    cf[0] = __uint_as_float(c[g * 8 + 2 * t]);
    cf[1] = __uint_as_float(c[g * 8 + 2 * t + 1]);
    cf[2] = __uint_as_float(c[(g + 8) * 8 + 2 * t]);
    cf[3] = __uint_as_float(c[(g + 8) * 8 + 2 * t + 1]);
}

__device__ __forceinline__ void store_d(uint32_t* d, const float df[4], int lane) {
    const int g = lane >> 2, t = lane & 3;
    d[g * 8 + 2 * t] = __float_as_uint(df[0]);
    d[g * 8 + 2 * t + 1] = __float_as_uint(df[1]);
    d[(g + 8) * 8 + 2 * t] = __float_as_uint(df[2]);
    d[(g + 8) * 8 + 2 * t + 1] = __float_as_uint(df[3]);
}

// One warp per tile: nvfp4 (ue4m3, 4 K-blocks of 16).
__global__ void nvfp4_tiles_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt, const uint8_t* __restrict__ SFA,
                                   const uint8_t* __restrict__ SFB, const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int ntiles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const int lane = threadIdx.x & 31;
    uint32_t a[4], b[2];
    pack_a_e2m1(a, A + (size_t)warp * 16 * 64, lane);
    pack_b_e2m1(b, Bt + (size_t)warp * 8 * 64, lane);
    const uint32_t sfa = pack_sfa(SFA + (size_t)warp * 16 * 4, 4, lane);
    const uint32_t sfb = pack_sfb(SFB + (size_t)warp * 8 * 4, 4, lane);
    float cf[4], df[4];
    load_c(cf, C + (size_t)warp * 128, lane);
    mma_nvfp4_m16n8k64(df, a, b, cf, sfa, sfb);
    store_d(D + (size_t)warp * 128, df, lane);
}

#ifndef NO_MXF4
// One warp per tile: mxfp4 (ue8m0, 2 K-blocks of 32).
__global__ void mxfp4_tiles_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt, const uint8_t* __restrict__ SFA,
                                   const uint8_t* __restrict__ SFB, const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int ntiles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const int lane = threadIdx.x & 31;
    uint32_t a[4], b[2];
    pack_a_e2m1(a, A + (size_t)warp * 16 * 64, lane);
    pack_b_e2m1(b, Bt + (size_t)warp * 8 * 64, lane);
    const uint32_t sfa = pack_sfa(SFA + (size_t)warp * 16 * 2, 2, lane);
    const uint32_t sfb = pack_sfb(SFB + (size_t)warp * 8 * 2, 2, lane);
    float cf[4], df[4];
    load_c(cf, C + (size_t)warp * 128, lane);
    mma_mxfp4_m16n8k64(df, a, b, cf, sfa, sfb);
    store_d(D + (size_t)warp * 128, df, lane);
}
#endif

// One warp per tile: unscaled e2m1 through kind::f8f6f4 (K = 32, byte containers).
__global__ void fp4raw_tiles_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt, const uint32_t* __restrict__ C,
                                    uint32_t* __restrict__ D, int ntiles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const int lane = threadIdx.x & 31;
    const uint8_t* At = A + (size_t)warp * 16 * 32;
    const uint8_t* Bw = Bt + (size_t)warp * 8 * 32;
    const int row0 = lane >> 2, kcol = 4 * (lane & 3);
    uint32_t a[4], b[2];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        const int row = row0 + 8 * (r & 1), k0 = kcol + 16 * (r >> 1);
        uint32_t v = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) v |= (uint32_t)((At[row * 32 + k0 + j] & 0xF) << 2) << (8 * j);
        a[r] = v;
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int k0 = kcol + 16 * r;
        uint32_t v = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) v |= (uint32_t)((Bw[row0 * 32 + k0 + j] & 0xF) << 2) << (8 * j);
        b[r] = v;
    }
    float cf[4], df[4];
    load_c(cf, C + (size_t)warp * 128, lane);
    mma_fp4raw_m16n8k32(df, a, b, cf);
    store_d(D + (size_t)warp * 128, df, lane);
}

// One warp per tile: bf16 m16n8k16 (identical to tools/tc_probe/mma_tiles.cu, rebuilt for sm_120).
__global__ void bf16_tiles_kernel(const uint16_t* __restrict__ A, const uint16_t* __restrict__ Bt, const uint32_t* __restrict__ C,
                                  uint32_t* __restrict__ D, int ntiles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    constexpr int K = 16;
    const uint16_t* a = A + (size_t)warp * 16 * K;
    const uint16_t* b = Bt + (size_t)warp * 8 * K;
    uint32_t af[4], bf[2];
    af[0] = *reinterpret_cast<const uint32_t*>(a + (g)*K + (t * 2));
    af[1] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * K + (t * 2));
    af[2] = *reinterpret_cast<const uint32_t*>(a + (g)*K + (t * 2 + K / 2));
    af[3] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * K + (t * 2 + K / 2));
    bf[0] = *reinterpret_cast<const uint32_t*>(b + g * K + (t * 2));
    bf[1] = *reinterpret_cast<const uint32_t*>(b + g * K + (t * 2 + K / 2));
    float cf[4], df[4];
    load_c(cf, C + (size_t)warp * 128, lane);
    mma_bf16_m16n8k16(df, af, bf, cf);
    store_d(D + (size_t)warp * 128, df, lane);
}

// Host-side plumbing shared by every entry point: upload, launch (4 warps per block), download, free.
struct Buf {
    void* p = nullptr;
    size_t n = 0;
};

static int run_tiles(int which, const void* A, size_t a_bytes, const void* Bt, size_t b_bytes, const void* SFA, size_t sfa_bytes,
                     const void* SFB, size_t sfb_bytes, const uint32_t* C, uint32_t* D, int ntiles) {
    const size_t c_bytes = (size_t)ntiles * 128 * 4;
    void *dA = nullptr, *dB = nullptr, *dSFA = nullptr, *dSFB = nullptr;
    uint32_t *dC = nullptr, *dD = nullptr;
    cudaError_t err = cudaSuccess;
    if ((err = cudaMalloc(&dA, a_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dB, b_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dSFA, sfa_bytes ? sfa_bytes : 1)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dSFB, sfb_bytes ? sfb_bytes : 1)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dC, c_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dD, c_bytes)) != cudaSuccess) return (int)err;
    cudaMemcpy(dA, A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bt, b_bytes, cudaMemcpyHostToDevice);
    if (sfa_bytes) cudaMemcpy(dSFA, SFA, sfa_bytes, cudaMemcpyHostToDevice);
    if (sfb_bytes) cudaMemcpy(dSFB, SFB, sfb_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, c_bytes, cudaMemcpyHostToDevice);
    const int wpb = 4, blocks = (ntiles + wpb - 1) / wpb;
    switch (which) {
        case 0: nvfp4_tiles_kernel<<<blocks, 32 * wpb>>>((const uint8_t*)dA, (const uint8_t*)dB, (const uint8_t*)dSFA, (const uint8_t*)dSFB, dC, dD, ntiles); break;
#ifndef NO_MXF4
        case 1: mxfp4_tiles_kernel<<<blocks, 32 * wpb>>>((const uint8_t*)dA, (const uint8_t*)dB, (const uint8_t*)dSFA, (const uint8_t*)dSFB, dC, dD, ntiles); break;
#endif
        case 2: fp4raw_tiles_kernel<<<blocks, 32 * wpb>>>((const uint8_t*)dA, (const uint8_t*)dB, dC, dD, ntiles); break;
        case 3: bf16_tiles_kernel<<<blocks, 32 * wpb>>>((const uint16_t*)dA, (const uint16_t*)dB, dC, dD, ntiles); break;
        default: err = cudaErrorInvalidValue;
    }
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) cudaMemcpy(D, dD, c_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dSFA);
    cudaFree(dSFB);
    cudaFree(dC);
    cudaFree(dD);
    return (int)err;
}

}  // namespace

extern "C" {

// A u8[n][16][64], Bt u8[n][8][64], SFA u8[n][16][4], SFB u8[n][8][4], C/D u32[n][16][8].  Returns the cudaError_t as an int.
int mma_nvfp4_tiles(const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB, const uint32_t* C, uint32_t* D, int n) {
    return run_tiles(0, A, (size_t)n * 16 * 64, Bt, (size_t)n * 8 * 64, SFA, (size_t)n * 16 * 4, SFB, (size_t)n * 8 * 4, C, D, n);
}

// A u8[n][16][64], Bt u8[n][8][64], SFA u8[n][16][2], SFB u8[n][8][2].  Returns -1 when compiled with -DNO_MXF4.
int mma_mxfp4_tiles(const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB, const uint32_t* C, uint32_t* D, int n) {
#ifdef NO_MXF4
    (void)A; (void)Bt; (void)SFA; (void)SFB; (void)C; (void)D; (void)n;
    return -1;
#else
    return run_tiles(1, A, (size_t)n * 16 * 64, Bt, (size_t)n * 8 * 64, SFA, (size_t)n * 16 * 2, SFB, (size_t)n * 8 * 2, C, D, n);
#endif
}

// A u8[n][16][32], Bt u8[n][8][32] (e2m1 codes, low nibble), C/D u32[n][16][8].
int mma_fp4raw_tiles(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int n) {
    return run_tiles(2, A, (size_t)n * 16 * 32, Bt, (size_t)n * 8 * 32, nullptr, 0, nullptr, 0, C, D, n);
}

// A u16[n][16][16], Bt u16[n][8][16] bf16 bits, C/D u32[n][16][8].
int mma_bf16_tiles(const uint16_t* A, const uint16_t* Bt, const uint32_t* C, uint32_t* D, int n) {
    return run_tiles(3, A, (size_t)n * 16 * 16 * 2, Bt, (size_t)n * 8 * 16 * 2, nullptr, 0, nullptr, 0, C, D, n);
}

int has_mxf4(void) {
#ifdef NO_MXF4
    return 0;
#else
    return 1;
#endif
}

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
