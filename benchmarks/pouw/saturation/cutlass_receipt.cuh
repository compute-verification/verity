// The interface between cutlass_receipt.cu (the gates and the timing) and its kernels, one cutlass_receipt_kernel.cu object
// each; no CUTLASS here.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

struct cr_args {
    int m, n, k;
    const void* A;               // m x k E4M3, row-major
    const void* B;               // n x k E4M3, row-major (the GEMM's k x n operand, column-major)
    const float *sa, *sb;        // per-row (token) and per-column (channel) scales
    void* D;                     // m x n bf16, row-major: bf16(sa[m] * (sb[n] * acc))
    uint32_t* R;                 // receipt kernels: one word a tile and MMA thread, its chains folded
    float* dump;                 // the dump kernel: every 32-deep k-block's prefix sums, [k / 32][m][n]
    uint32_t* map;               // the dump kernel: (row << 16 | col) within the tile of each thread's chain words
};

struct cr_impl {
    virtual int launch(cudaStream_t s) = 0;
    virtual ~cr_impl() {}
};

struct cr_kernel {
    const char* name;
    int cfg;                     // 0: vLLM's prefill kernel (cooperative 128x128x128), 1: its decode kernel (ping-pong 32x64x128)
    int var, rounds, p, mode;    // phi.cuh's VAR and ROUNDS, L = 128 p; mode 0 plain, 1 receipt, 2 dump, 3 extra only
    // spread 0: every chain after the k-tile's last k-block. spread 1: chain (mm, nn) after k-block nn * kblocks / nn_max
    // of the k-tile, its prefix at that k-block's end. grain g > 1 (p 1, spread 0): every chain after k-blocks
    // kblocks / g - 1, 2 kblocks / g - 1, ..., with w the k-block's index from 1; L = 128 p / grain.
    int spread, grain;
    int extra;                   // a cheater's own alu phi1 steps a thread after every 32-deep k-block (mode 3: no receipt)
    int honest;                  // the first honest chain words (ch * 4 + e) of each receipt point get rounds + 1
    int forced;                  // forced alu phi1 steps a thread after every 32-deep k-block that read no accumulator
    int tile_m, tile_n, threads, chains, nn_max, kblocks, stages, smem_bytes;
    cr_impl* (*make)(const cr_args& a, int* err);
};

inline std::vector<const cr_kernel*>& cr_registry() {
    static std::vector<const cr_kernel*> r;
    return r;
}
struct cr_register {
    explicit cr_register(const cr_kernel* k) { cr_registry().push_back(k); }
};
