// Single-instruction tensor-core probes for Hopper `wgmma`: exactly one instruction per warpgroup per tile (sm_90a only).
//
//   wgmma_e4m3_m64n8k32_tiles   wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3   A 64x32 e4m3, Bt 8x32 e4m3, C/D 64x8 f32
//   wgmma_e5m2_m64n8k32_tiles   wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2   A 64x32 e5m2, Bt 8x32 e5m2, C/D 64x8 f32
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

// The descriptor offsets of the current call (set by the extern "C" entry points before run_tiles launches).
uint32_t g_lbo = 128, g_sbo = 256;

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

}  // extern "C"
