// cuda/sha512.cuh's launch_hm96_finish, for test_hm96_leaves.py: hm96_leaves finishes leaves [base, n) of n digests, hashing
// at blocks of tpb threads as launch_merkle512 (256) and flock_cuda_hm96_leaves (128) do, leaving the digests below base, and
// refuses a tpb hm96_tpb_fits refuses. benchmarks/flock_kernels/hm96_leaves.cu times it. hm96_trees builds whole salted trees
// through launch_merkle512, their salts copied ahead of the builds or beside them.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../cuda/sha512.cuh"

extern "C" int hm96_leaves(const uint8_t* x, const uint8_t* salts, long long n, long long base, int tpb, const uint8_t* prefixes,
                           const uint64_t* tables, uint8_t* out) {
    if (n < 1 || base < 0 || base >= n || !hm96_tpb_fits(tpb)) return 1;
    std::vector<uint32_t> frag(2 * HM96_FRAGS);
    hm96_fragments(tables, frag.data());
    uint8_t *d_x = nullptr, *d_s = nullptr, *d_pre = nullptr;
    uint32_t* d_f = nullptr;
    uint64_t* d_mid = nullptr;
    cudaError_t e = cudaMalloc(&d_x, (size_t)n * 64);
    if (e == cudaSuccess) e = cudaMalloc(&d_s, (size_t)(n - base) * 192);
    if (e == cudaSuccess) e = cudaMalloc(&d_pre, 256);
    if (e == cudaSuccess) e = cudaMalloc(&d_f, frag.size() * sizeof(uint32_t));
    if (e == cudaSuccess) e = cudaMalloc(&d_mid, 16 * sizeof(uint64_t));
    if (e == cudaSuccess) e = cudaMemcpy(d_x, x, (size_t)n * 64, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_s, salts, (size_t)(n - base) * 192, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_pre, prefixes, 256, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_f, frag.data(), frag.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (e == cudaSuccess) {
        hm96_midstates<<<1, 1>>>(d_pre, d_mid);
        launch_hm96_finish(d_x, (const uint32_t*)d_s, base, n, tpb, d_f, d_mid);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemcpy(out, d_x, (size_t)n * 64, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_s); cudaFree(d_pre); cudaFree(d_f); cudaFree(d_mid);
    return e == cudaSuccess ? 0 : 2;
}

struct TreeSalts {
    const uint8_t* salts;
    const long long* at;
    int calls;
};

// the j-th call's tree takes the salts from leaf at[j] on; only the first call is tree 0's
static const uint8_t* tree_salts(void* ctx, int tree, long long) {
    TreeSalts* t = (TreeSalts*)ctx;
    const int j = t->calls++;
    return (tree == 0) == (j == 0) ? t->salts + 192 * t->at[j] : nullptr;
}

// launch_merkle512's salted trees under an Hm96Guard: tree j of leaves[j] leaves over x's first leaves[j] leaves of leaf_size
// bytes, tree 0 level 0's, each tree's salts the next leaves[j] of `salts`, built in order with trees 1.. copied ahead of
// their builds (hm96_copy_ahead) when `ahead` is set, else each beside its own build; every tree's 2 leaves[j] - 1 nodes out
// in order. 3: the copies ahead refused, 4: a tree copied ahead and never built.
extern "C" int hm96_trees(const uint8_t* x, int leaf_size, const long long* leaves, int trees, const uint8_t* salts, int ahead,
                          const uint8_t* prefixes, const uint64_t* tables, uint8_t* out) {
    if (trees < 1) return 1;
    std::vector<long long> at(trees, 0);
    long long most = 0;
    for (int j = 0; j < trees; j++) {
        if (j) at[j] = at[j - 1] + leaves[j - 1];
        if (leaves[j] > most) most = leaves[j];
    }
    TreeSalts ts{salts, at.data(), 0};
    uint8_t* d_x = nullptr;
    std::vector<uint8_t*> d_tree(trees, nullptr), d_s(trees, nullptr);
    cudaError_t e = cudaMalloc(&d_x, (size_t)most * leaf_size);
    if (e == cudaSuccess) e = cudaMemcpy(d_x, x, (size_t)most * leaf_size, cudaMemcpyHostToDevice);
    for (int j = 0; j < trees && e == cudaSuccess; j++) {
        e = cudaMalloc(&d_tree[j], (size_t)(2 * leaves[j] - 1) * 64);
        if (e == cudaSuccess && j) e = cudaMalloc(&d_s[j], (size_t)leaves[j] * 192);
    }
    int rc = e == cudaSuccess ? 0 : 2;
    if (!rc) {
        Hm96Guard guard(tree_salts, &ts, prefixes, tables, leaves[0]);
        launch_merkle512(d_x, d_tree[0], leaves[0], leaf_size);
        if (ahead && !hm96_copy_ahead(std::vector<long long>(leaves + 1, leaves + trees),
                                      std::vector<uint8_t*>(d_s.begin() + 1, d_s.end())))
            rc = 3;
        for (int j = 1; j < trees; j++) launch_merkle512(d_x, d_tree[j], leaves[j], leaf_size);
        if (!g_hm96.ahead.empty()) rc = 4;
        e = cudaDeviceSynchronize();
        long long o = 0;
        for (int j = 0; j < trees && e == cudaSuccess; j++) {
            e = cudaMemcpy(out + (size_t)o * 64, d_tree[j], (size_t)(2 * leaves[j] - 1) * 64, cudaMemcpyDeviceToHost);
            o += 2 * leaves[j] - 1;
        }
        if (!rc && e != cudaSuccess) rc = 2;
    }
    cudaFree(d_x);
    for (int j = 0; j < trees; j++) { cudaFree(d_tree[j]); cudaFree(d_s[j]); }
    return rc;
}
