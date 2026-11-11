// test_async_copies.py's checks of cuda/async_copies.cuh: ac_check returns a bitmask of the checks that failed (0: all pass).
#include <cstdint>
#include <cstring>
#include <vector>
#include "async_copies.cuh"

__constant__ uint64_t ac_symbol[8];

__global__ void ac_fill(uint64_t* p, size_t n, uint64_t v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v + i;
}

// a long kernel before a copy, so that a copy that didn't wait behind it would find the old value
__global__ void ac_slow_fill(uint64_t* p, uint64_t v, long long rounds) {
    uint64_t x = v;
    for (long long i = 0; i < rounds; i++) x = x * 6364136223846793005ull + 1442695040888963407ull;
    p[1] = x;
    p[0] = v;
}

__global__ void ac_read_symbol(uint64_t* out) { out[threadIdx.x] = ac_symbol[threadIdx.x]; }

static bool ac_ramp(const std::vector<uint64_t>& h, uint64_t v) {
    for (size_t i = 0; i < h.size(); i++) if (h[i] != v + i) return false;
    return true;
}

extern "C" int ac_check() {
    int bad = 0;
    uint64_t* d;
    const size_t big = HC_DIRECT / 8 + 1, n = 4 * HC_STAGE / 8;
    if (cudaMalloc(&d, n * 8) != cudaSuccess) return -1;

    // 1: reads come back in stream order, each the value its stream position saw, and only at hc_wait
    {
        uint64_t a = 0, b = 0;
        ac_slow_fill<<<1, 1>>>(d, 11, 1 << 22);
        hc_get(&a, d, 8);
        ac_fill<<<1, 1>>>(d, 1, 22);
        hc_get(&b, d, 8);
        const bool untouched = a == 0 && b == 0;
        if (hc_wait() != cudaSuccess || !untouched || a != 11 || b != 22) bad |= 1;
    }
    // 2: an upload takes its bytes at the call: the caller's buffer may change at once
    {
        std::vector<uint64_t> h(64), back(64);
        for (size_t i = 0; i < h.size(); i++) h[i] = 100 + i;
        ac_slow_fill<<<1, 1>>>(d + 1000, 0, 1 << 22);
        hc_put(d, h.data(), h.size() * 8);
        std::fill(h.begin(), h.end(), 0);
        hc_get(back.data(), d, back.size() * 8);
        if (hc_wait() != cudaSuccess || !ac_ramp(back, 100)) bad |= 2;
    }
    // 3: uploads past the staging's size wrap after a wait, each still the bytes given
    {
        const size_t words = HC_DIRECT / 8 / 2, puts = 3 * HC_STAGE / HC_DIRECT;
        std::vector<uint64_t> h(words), back(words);
        bool ok = true;
        for (size_t r = 0; r < puts; r++) {
            for (size_t i = 0; i < words; i++) h[i] = 1000000000ull * r + i;
            hc_put(d + r * words, h.data(), words * 8);
        }
        for (size_t r = 0; r < puts && ok; r++) {
            hc_get(back.data(), d + r * words, words * 8);
            ok = hc_wait() == cudaSuccess && ac_ramp(back, 1000000000ull * r);
        }
        if (!ok) bad |= 4;
    }
    // 4: a copy past HC_DIRECT goes through cudaMemcpy, in order with the queued ones
    {
        std::vector<uint64_t> h(big), back(big);
        for (size_t i = 0; i < big; i++) h[i] = 7 + i;
        uint64_t small = 0;
        ac_fill<<<64, 256>>>(d, 1, 5);
        hc_get(&small, d, 8);
        hc_put(d, h.data(), big * 8);
        hc_get(back.data(), d, big * 8);
        if (hc_wait() != cudaSuccess || small != 5 || !ac_ramp(back, 7)) bad |= 8;
    }
    // 5: a symbol upload lands before the kernels queued after it, behind the ones queued before it
    {
        uint64_t h[8];
        uint64_t* d_out = d + 64;
        for (int r = 0; r < 3; r++) {
            for (int i = 0; i < 8; i++) h[i] = 50 * r + i;
            hc_put_symbol(ac_symbol, h, sizeof(h));
            ac_read_symbol<<<1, 8>>>(d_out + 8 * r);
        }
        std::vector<uint64_t> back(24);
        hc_get(back.data(), d_out, 24 * 8);
        bool ok = hc_wait() == cudaSuccess;
        for (int r = 0; r < 3 && ok; r++)
            for (int i = 0; i < 8; i++) ok = ok && back[8 * r + i] == (uint64_t)(50 * r + i);
        if (!ok) bad |= 16;
    }
    // 6: hc_drop completes the queued copies and leaves the reads' destinations alone
    {
        uint64_t a = 1;
        ac_fill<<<1, 1>>>(d, 1, 9);
        hc_get(&a, d, 8);
        hc_drop();
        uint64_t b = 0;
        hc_get(&b, d, 8);
        if (hc_wait() != cudaSuccess || a != 1 || b != 9) bad |= 32;
    }
    cudaFree(d);
    return bad;
}
