// hiding_gpu_cost.py's SHA-512 alternative: C-Flock's SHA-512 leaf, hm96-sha512/v1 and node kernels (backends/flock/cuda/sha512.cuh)
// over one step of 256-byte chunks, timed by phase (medians of `reps` after a warm-up, CUDA events).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <functional>
#include <vector>
#include "sha512.cuh"

torch::Tensor bench(torch::Tensor data, torch::Tensor tree, int64_t n, int64_t leaf_size, torch::Tensor tables, torch::Tensor prefixes,
                    torch::Tensor key, int64_t reps) {
    TORCH_CHECK(data.is_cuda() && tree.is_cuda() && tables.is_cuda() && prefixes.is_cuda() && key.is_cuda());
    TORCH_CHECK(tree.numel() >= 2 * n * 64 && tables.numel() == 192 * 256 * 64 && prefixes.numel() == 256 && key.numel() == 32);
    uint64_t *mid = nullptr;
    cudaMalloc(&mid, 16 * sizeof(uint64_t));
    hm96_midstates<<<1, 1>>>(prefixes.data_ptr<uint8_t>(), mid);
    Hm96Key hk{};
    cudaMemcpy(hk.k, key.data_ptr<uint8_t>(), 32, cudaMemcpyDeviceToHost);
    const uint8_t *d = data.data_ptr<uint8_t>();
    uint8_t *t = tree.data_ptr<uint8_t>();
    const uint64_t *tb = reinterpret_cast<const uint64_t *>(tables.data_ptr<uint8_t>());
    const int tpb = 256;
    auto leaves = [&] {
        int wpb = tpb >> 5;
        long long warps = (n + 31) / 32, blocks = (warps + wpb - 1) / wpb;
        sha512_staged_merkle_leaves<<<(unsigned)blocks, tpb, (size_t)wpb * 32 * 33 * sizeof(uint32_t)>>>(d, t, n, (int)leaf_size);
    };
    auto hm = [&] { hm96_finish_leaves<<<(unsigned)((n + tpb - 1) / tpb), tpb>>>(t, hk, 1, tb, n, mid); };
    auto nodes = [&] {
        long long rs = 0, rl = n;
        while (rl > 1) {
            long long next = rl >> 1, ws = rs + rl;
            sha512_merkle_level<<<(unsigned)((next + tpb - 1) / tpb), tpb>>>(t, rs, ws, next);
            rs += rl;
            rl = next;
        }
    };
    std::vector<std::vector<float>> ms(3);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int r = 0; r <= reps; ++r) {
        int k = 0;
        for (auto f : {std::function<void()>(leaves), std::function<void()>(hm), std::function<void()>(nodes)}) {
            cudaEventRecord(a);
            f();
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float x = 0;
            cudaEventElapsedTime(&x, a, b);
            if (r) ms[k].push_back(x);
            ++k;
        }
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(mid);
    auto out = torch::empty({3}, torch::kFloat64);
    for (int k = 0; k < 3; ++k) {
        std::sort(ms[k].begin(), ms[k].end());
        out[k] = ms[k][ms[k].size() / 2];
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("bench", &bench, "SHA-512 chunk leaves, hm96-sha512 leaves and tree, timed"); }
