// `tanh.approx.f32` (MUFU.TANH, what FA2's `apply_softcap` runs through cutlass::fast_tanh) on every 32-bit input, against the
// registered `MufuTanh_v1` (prims.py): its three exhaustively scanned rules and its measured table.
//
//   tanh_probe <ref.bin> <lo> <hi> <ident> <sat> [out.bin]
//     ref.bin : u32 LE words, tanh.approx.f32 of every word in [lo, hi) as MufuTanh_v1 has them (the in-tree table)
//     out.bin : this device's words on [lo, hi), the table another MufuTanh version would pin
//     rules   : |x| < ident -> x (bitwise, both signs); ident <= |x| <= +inf and |x| >= sat -> +-1.0; NaN (any payload) -> 0x7FFFFFFF
//   Every |x| word is evaluated with both signs; prints one JSON line: per class the words checked and the mismatches, the odd-symmetry
//   violations (tanh(-x) != -tanh(x) on non-NaN words), and the first mismatching inputs.
//
// Build as FA2 is shipped (compute_80 PTX, JIT-compiled by the driver): nvcc -O2 -gencode arch=compute_80,code=compute_80 -o tanh_probe tanh_probe.cu

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __forceinline__ uint32_t tanh_approx(uint32_t u) {
    float y;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(__uint_as_float(u)));
    return __float_as_uint(y);
}

enum { C_NAN, C_SAT, C_IDENT, C_TABLE, C_SYM, C_N };
#define FIRST 8

__global__ void k_probe(const uint32_t* ref, uint32_t lo, uint32_t hi, uint32_t ident, uint32_t sat,
                        unsigned long long* checked, unsigned long long* bad, uint32_t* first, unsigned int* nfirst, uint32_t* meas) {
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (1ull << 31); i += stride) {
        uint32_t a = (uint32_t)i;
        uint32_t yp = tanh_approx(a), yn = tanh_approx(a | 0x80000000u);
        if (a >= lo && a < hi) meas[a - lo] = yp;
        int c;
        bool ok;
        if (a > 0x7F800000u) {
            c = C_NAN; ok = yp == 0x7FFFFFFFu && yn == 0x7FFFFFFFu;
        } else if (a >= sat) {
            c = C_SAT; ok = yp == 0x3F800000u && yn == 0xBF800000u;
        } else if (a < ident) {
            c = C_IDENT; ok = yp == a && yn == (a | 0x80000000u);
        } else if (a >= lo && a < hi) {
            c = C_TABLE; ok = yp == ref[a - lo] && yn == (ref[a - lo] ^ 0x80000000u);
        } else {
            c = C_TABLE; ok = false;
        }
        atomicAdd(&checked[c], 2ull);
        if (!ok) {
            atomicAdd(&bad[c], 1ull);
            unsigned int k = atomicAdd(nfirst, 1u);
            if (k < FIRST) { first[3 * k] = a; first[3 * k + 1] = yp; first[3 * k + 2] = yn; }
        }
        if (a <= 0x7F800000u) {
            atomicAdd(&checked[C_SYM], 1ull);
            if (yn != (yp ^ 0x80000000u)) atomicAdd(&bad[C_SYM], 1ull);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) { fprintf(stderr, "usage: %s ref.bin lo hi ident sat [out.bin]\n", argv[0]); return 2; }
    uint32_t lo = strtoul(argv[2], 0, 0), hi = strtoul(argv[3], 0, 0), ident = strtoul(argv[4], 0, 0), sat = strtoul(argv[5], 0, 0);
    size_t n = hi - lo;
    std::vector<uint32_t> h(n);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(h.data(), 4, n, f) != n) { fprintf(stderr, "cannot read %zu words from %s\n", n, argv[1]); return 2; }
    fclose(f);
    uint32_t *ref, *first, *meas;
    unsigned long long *checked, *bad;
    unsigned int* nfirst;
    cudaMalloc(&ref, n * 4);
    cudaMemcpy(ref, h.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&checked, C_N * 8); cudaMemset(checked, 0, C_N * 8);
    cudaMalloc(&bad, C_N * 8); cudaMemset(bad, 0, C_N * 8);
    cudaMalloc(&first, 3 * FIRST * 4); cudaMemset(first, 0, 3 * FIRST * 4);
    cudaMalloc(&nfirst, 4); cudaMemset(nfirst, 0, 4);
    cudaMalloc(&meas, n * 4);
    k_probe<<<4096, 256>>>(ref, lo, hi, ident, sat, checked, bad, first, nfirst, meas);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); return 3; }
    unsigned long long hc[C_N], hb[C_N];
    uint32_t hf[3 * FIRST];
    unsigned int hn;
    cudaMemcpy(hc, checked, sizeof hc, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, bad, sizeof hb, cudaMemcpyDeviceToHost);
    cudaMemcpy(hf, first, sizeof hf, cudaMemcpyDeviceToHost);
    cudaMemcpy(&hn, nfirst, 4, cudaMemcpyDeviceToHost);
    if (argc == 7) {
        cudaMemcpy(h.data(), meas, n * 4, cudaMemcpyDeviceToHost);
        FILE* o = fopen(argv[6], "wb");
        if (!o || fwrite(h.data(), 4, n, o) != n) { fprintf(stderr, "cannot write %s\n", argv[6]); return 2; }
        fclose(o);
    }
    const char* names[C_N] = {"nan", "saturate", "identity", "table", "odd_symmetry"};
    printf("{");
    for (int c = 0; c < C_N; c++) printf("\"%s\": {\"checked\": %llu, \"mismatches\": %llu}, ", names[c], hc[c], hb[c]);
    printf("\"first\": [");
    for (unsigned int k = 0; k < hn && k < FIRST; k++) printf("%s[\"0x%08X\", \"0x%08X\", \"0x%08X\"]", k ? ", " : "", hf[3 * k], hf[3 * k + 1], hf[3 * k + 2]);
    printf("]}\n");
    return 0;
}
