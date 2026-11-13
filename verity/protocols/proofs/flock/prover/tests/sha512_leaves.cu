// cuda/sha512.cuh's leaf hashing on the device, for test_sha512_leaves.py. flock_tree hashes host leaves placed at a byte
// offset into a device buffer (offset 8: unaligned) by one path: 0 launch_merkle512 (the whole tree, 2n - 1 digests for
// n a power of two), 1 sha512_merkle_leaves_bytes or 2 sha512_staged_merkle_leaves (the n leaves).
// Built with -DSHA512_LEAVES_MAIN it is also `sha512_leaves N LEAF REPS`: N leaves (a power of two) of LEAF bytes (LEAF % 16
// == 0) filled on the device from a fixed seed, and one JSON line: launch_merkle512's root, the mean time over REPS launches of
// launch_merkle512 and of each leaf kernel, and the leaves where the staged kernel's digest differs from the byte kernel's.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../cuda/sha512.cuh"

static void launch_leaves(int path, const uint8_t* d, uint8_t* tree, long long n, int leaf_size) {
    const int tpb = 256, wpb = tpb >> 5;
    if (path == 1) {
        sha512_merkle_leaves_bytes<<<(unsigned)((n + tpb - 1) / tpb), tpb>>>(d, tree, n, leaf_size);
    } else {
        const long long blocks = ((n + 31) / 32 + wpb - 1) / wpb;
        sha512_staged_merkle_leaves<<<(unsigned)blocks, tpb, (size_t)wpb * 32 * 33 * sizeof(uint32_t)>>>(d, tree, n, leaf_size);
    }
}

extern "C" int flock_tree(int path, const uint8_t* data, uint64_t num_leaves, int leaf_size, int offset, uint8_t* out) {
    const long long n = (long long)num_leaves;
    const size_t bytes = (size_t)n * leaf_size, digests = path == 0 ? 2 * n - 1 : n;
    uint8_t *buf = nullptr, *tree = nullptr;
    cudaError_t e = cudaMalloc(&buf, bytes + offset);
    if (e == cudaSuccess) e = cudaMalloc(&tree, digests * 64);
    if (e == cudaSuccess) e = cudaMemcpy(buf + offset, data, bytes, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) {
        if (path == 0) launch_merkle512(buf + offset, tree, n, leaf_size);
        else launch_leaves(path, buf + offset, tree, n, leaf_size);
        e = cudaDeviceSynchronize();
    }
    if (e == cudaSuccess) e = cudaMemcpy(out, tree, digests * 64, cudaMemcpyDeviceToHost);
    cudaFree(buf);
    cudaFree(tree);
    return (int)e;
}

#ifdef SHA512_LEAVES_MAIN
__global__ void fill_words(uint64_t* d, size_t words) {   // splitmix64 of the word's index
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= words) return;
    uint64_t z = (i + 1) * 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    d[i] = z ^ (z >> 31);
}

__global__ void count_differing(const uint4* a, const uint4* b, long long n, unsigned long long* bad) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool same = true;
    for (int q = 0; q < 4; q++) {
        const uint4 x = a[4 * i + q], y = b[4 * i + q];
        same &= x.x == y.x && x.y == y.y && x.z == y.z && x.w == y.w;
    }
    if (!same) atomicAdd(bad, 1ull);
}

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: sha512_leaves N LEAF REPS\n"); return 2; }
    const long long n = atoll(argv[1]);
    const int leaf_size = atoi(argv[2]), reps = atoi(argv[3]);
    if (n < 1 || leaf_size < 16 || leaf_size % 16 || reps < 1) { fprintf(stderr, "N >= 1, LEAF a multiple of 16, REPS >= 1\n"); return 2; }
    const size_t bytes = (size_t)n * leaf_size;
    uint8_t *data, *tree, *ref;
    unsigned long long* bad;
    CHECK(cudaMalloc(&data, bytes));
    CHECK(cudaMalloc(&tree, (size_t)(2 * n - 1) * 64));
    CHECK(cudaMalloc(&ref, (size_t)n * 64));
    CHECK(cudaMalloc(&bad, sizeof(*bad)));
    fill_words<<<(unsigned)((bytes / 8 + 255) / 256), 256>>>((uint64_t*)data, bytes / 8);
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    float ms[3];
    for (int p = 0; p < 3; p++) {   // 0 launch_merkle512, 1 the byte kernel, 2 the staged kernel
        uint8_t* out = p == 1 ? ref : tree;
        if (p == 0) launch_merkle512(data, out, n, leaf_size); else launch_leaves(p, data, out, n, leaf_size);
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < reps; r++) {
            if (p == 0) launch_merkle512(data, out, n, leaf_size); else launch_leaves(p, data, out, n, leaf_size);
        }
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        CHECK(cudaGetLastError());
        CHECK(cudaEventElapsedTime(&ms[p], t0, t1));
        ms[p] /= reps;
        if (p == 0) {
            uint8_t root[64];
            CHECK(cudaMemcpy(root, tree + (size_t)(2 * n - 2) * 64, 64, cudaMemcpyDeviceToHost));
            printf("{\"leaves\": %lld, \"leaf_size\": %d, \"reps\": %d, \"root\": \"", n, leaf_size, reps);
            for (int i = 0; i < 64; i++) printf("%02x", root[i]);
            printf("\"");
        }
    }
    CHECK(cudaMemset(bad, 0, sizeof(*bad)));
    count_differing<<<(unsigned)((n + 255) / 256), 256>>>((const uint4*)tree, (const uint4*)ref, n, bad);
    unsigned long long nbad = 0;
    CHECK(cudaMemcpy(&nbad, bad, sizeof(nbad), cudaMemcpyDeviceToHost));
    printf(", \"tree_ms\": %.4f, \"bytes_leaves_ms\": %.4f, \"staged_leaves_ms\": %.4f, \"staged_differs_from_bytes\": %llu}\n",
           ms[0], ms[1], ms[2], nbad);
    return nbad != 0;
}
#endif
