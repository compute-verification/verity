// One integer `mma.sync` per warp per tile on sm_120: m16n8k32 with s8 or u8 operands and an s32 accumulator, with and
// without `.satfinite`, eight forms in all.  Every output word is the instruction's own function of (C[i][j], A[i][:],
// Bt[j][:]): no software reduction in between, and operands and results move as raw bits.
//
//   imma_probe gpu CASES OUTDIR    runs the eight forms over CASES (imma_probe.py gen) and writes OUTDIR/d_<form>.bin
//
// A tile is A[16][32] and Bt[8][32] operand bytes (k contiguous; Bt is the ".col" B operand) and C[16][8] s32 words.  The
// fragment layout is benchmarks/tc_probe/mma_tiles.cu's for 8-bit m16n8k32 (the PTX ISA's "mma.m16n8k32" figures): with
// g = lane>>2 and t = lane&3, A register i holds row g (+8 for odd i), k = 4t..4t+3 (+16 for i >= 2); B register i holds
// column g, the same k range.  imma_probe.py's random tiles check the layout: a wrong one fails them all.
//
// Build: nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o imma_probe imma_probe.cu
// Each form is its own extern "C" kernel, so cuobjdump -sass shows each one's single MMA-class instruction by name.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#define IMMA_FORM(NAME, PTX)                                                                                                \
    extern "C" __global__ void NAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,                          \
                                    const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int ntiles) {                 \
        const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                                      \
        if (warp >= ntiles) return;                                                                                         \
        const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;                                                     \
        const uint8_t* a = A + (size_t)warp * 512;                                                                          \
        const uint8_t* b = Bt + (size_t)warp * 256;                                                                         \
        const uint32_t* c = C + (size_t)warp * 128;                                                                         \
        uint32_t* d = D + (size_t)warp * 128;                                                                               \
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(a + g * 32 + 4 * t);                                         \
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(a + (g + 8) * 32 + 4 * t);                                   \
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(a + g * 32 + 16 + 4 * t);                                    \
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(a + (g + 8) * 32 + 16 + 4 * t);                              \
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b + g * 32 + 4 * t);                                         \
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(b + g * 32 + 16 + 4 * t);                                    \
        const uint32_t c0 = c[g * 8 + 2 * t], c1 = c[g * 8 + 2 * t + 1];                                                    \
        const uint32_t c2 = c[(g + 8) * 8 + 2 * t], c3 = c[(g + 8) * 8 + 2 * t + 1];                                        \
        uint32_t d0, d1, d2, d3;                                                                                            \
        asm volatile(PTX " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                    \
                     : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)                                                               \
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));           \
        d[g * 8 + 2 * t] = d0;                                                                                              \
        d[g * 8 + 2 * t + 1] = d1;                                                                                          \
        d[(g + 8) * 8 + 2 * t] = d2;                                                                                        \
        d[(g + 8) * 8 + 2 * t + 1] = d3;                                                                                    \
    }

IMMA_FORM(imma_s8s8_sat, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32")
IMMA_FORM(imma_s8u8_sat, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.u8.s32")
IMMA_FORM(imma_u8s8_sat, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32")
IMMA_FORM(imma_u8u8_sat, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32")
IMMA_FORM(imma_s8s8_wrap, "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
IMMA_FORM(imma_s8u8_wrap, "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32")
IMMA_FORM(imma_u8s8_wrap, "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32")
IMMA_FORM(imma_u8u8_wrap, "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32")

namespace {

typedef void (*kernel_fn)(const uint8_t*, const uint8_t*, const uint32_t*, uint32_t*, int);

struct Form {
    const char* name;
    kernel_fn kernel;
};

const Form FORMS[] = {
    {"s8s8_sat", imma_s8s8_sat},   {"s8u8_sat", imma_s8u8_sat},   {"u8s8_sat", imma_u8s8_sat},   {"u8u8_sat", imma_u8u8_sat},
    {"s8s8_wrap", imma_s8s8_wrap}, {"s8u8_wrap", imma_s8u8_wrap}, {"u8s8_wrap", imma_u8s8_wrap}, {"u8u8_wrap", imma_u8u8_wrap},
};

constexpr int WARPS_PER_BLOCK = 4;

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
    if (argc != 4 || strcmp(argv[1], "gpu") != 0) {
        fprintf(stderr, "usage: imma_probe gpu CASES OUTDIR\n");
        return 2;
    }
    // CASES: "IMMAPRB1", u32 ntiles, u32 0, then A (ntiles*512 bytes), Bt (ntiles*256), C (ntiles*128 u32, little-endian).
    const std::vector<uint8_t> cases = read_all(argv[2]);
    if (cases.size() < 16 || memcmp(cases.data(), "IMMAPRB1", 8) != 0) {
        fprintf(stderr, "%s is not an imma_probe case file\n", argv[2]);
        return 2;
    }
    uint32_t ntiles;
    memcpy(&ntiles, cases.data() + 8, 4);
    const size_t a_bytes = (size_t)ntiles * 512, b_bytes = (size_t)ntiles * 256, c_bytes = (size_t)ntiles * 512;
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
    std::vector<uint32_t> host((size_t)ntiles * 128);
    const int blocks = (int)((ntiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    for (const Form& form : FORMS) {
        check(cudaMemset(dD, 0xA5, c_bytes), "memset D");  // a word the kernel never wrote reads 0xA5A5A5A5
        form.kernel<<<blocks, 32 * WARPS_PER_BLOCK>>>(dA, dB, dC, dD, (int)ntiles);
        check(cudaGetLastError(), form.name);
        check(cudaDeviceSynchronize(), form.name);
        check(cudaMemcpy(host.data(), dD, c_bytes, cudaMemcpyDeviceToHost), "copy D");
        write_all(out + "/d_" + form.name + ".bin", host.data(), c_bytes);
        printf("%s: %u tiles\n", form.name, ntiles);
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);
    return 0;
}
