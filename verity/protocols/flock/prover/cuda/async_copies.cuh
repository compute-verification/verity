// verity flock: copies between the host and the default stream that hold the host only where it reads a value.
// Installed into cuda-ghash by verity/protocols/flock/prover/cuda_chunk_patch.py; the provers' headers include it.
//
// hc_put queues an upload: its bytes go into mapped page-locked staging at once (the caller's buffer may change or go as soon
// as it returns) and a kernel on the stream copies them, so the host doesn't wait for the kernels queued before it. hc_get
// queues a kernel that copies into staging; hc_wait synchronizes the stream once and moves every queued read into its
// destination, which must stay valid until then. A transcript read is its hc_gets then one hc_wait; a read the host doesn't
// need yet (an opening's rows and paths) waits in the queue for the next transcript read. The copies are kernels on the
// stream, not cudaMemcpyAsync, with which the next session's witness, built beside this one on another thread, kept its
// session waiting longer. Staging is reused only after an hc_wait, which completes every copy queued before it: a copy that
// doesn't fit waits first, and one larger than HC_DIRECT goes through cudaMemcpy. The state is the calling thread's.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

static constexpr size_t HC_STAGE = (size_t)16 << 20;   // staging bytes, allocated on a thread's first copy
static constexpr size_t HC_DIRECT = (size_t)4 << 20;   // copies past this many bytes are cudaMemcpy

struct HcRead { void* dst; size_t at, bytes; };
struct HostCopies {
    uint8_t* stage = nullptr;   // the host's view of the staging
    uint8_t* dev = nullptr;     // the device's
    size_t used = 0;
    std::vector<HcRead> reads;
};
static thread_local HostCopies hc_state;

static __global__ void hc_copy_kernel(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    if ((((uintptr_t)dst | (uintptr_t)src | n) & 15) == 0) {
        uint4* d = (uint4*)dst;
        const uint4* s = (const uint4*)src;
        for (size_t k = i; k < n / 16; k += stride) d[k] = s[k];
    } else {
        for (size_t k = i; k < n; k += stride) dst[k] = src[k];
    }
}

inline cudaError_t hc_copy(void* dst, const void* src, size_t bytes) {
    const size_t units = (bytes + 15) / 16;
    const unsigned blocks = (unsigned)std::min<size_t>((units + 255) / 256, 1024);
    hc_copy_kernel<<<blocks, 256, 0, 0>>>((uint8_t*)dst, (const uint8_t*)src, bytes);
    return cudaGetLastError();
}

// every queued copy done and every queued read in its destination
inline cudaError_t hc_wait() {
    HostCopies& h = hc_state;
    const cudaError_t e = cudaStreamSynchronize(0);
    if (e == cudaSuccess)
        for (const HcRead& r : h.reads) memcpy(r.dst, h.stage + r.at, r.bytes);
    h.reads.clear();
    h.used = 0;
    return e;
}

// every queued copy done and the queued reads dropped: an error return's way out, where a destination may be gone
inline void hc_drop() {
    HostCopies& h = hc_state;
    if (!h.reads.empty() || h.used) cudaStreamSynchronize(0);
    h.reads.clear();
    h.used = 0;
}

// A prove call's guard: on any return the queued reads are dropped (a successful call has waited for them already)
struct HcDrop {
    ~HcDrop() { hc_drop(); }
};

inline cudaError_t hc_room(size_t bytes, size_t* at) {
    HostCopies& h = hc_state;
    if (!h.stage) {
        if (cudaHostAlloc((void**)&h.stage, HC_STAGE, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            h.stage = nullptr;
            return cudaErrorMemoryAllocation;
        }
        if (cudaHostGetDevicePointer((void**)&h.dev, h.stage, 0) != cudaSuccess) {
            cudaFreeHost(h.stage);
            h.stage = h.dev = nullptr;
            return cudaErrorMemoryAllocation;
        }
    }
    size_t a = (h.used + 63) & ~(size_t)63;
    if (a + bytes > HC_STAGE) {
        const cudaError_t e = hc_wait();
        if (e != cudaSuccess) return e;
        a = 0;
    }
    *at = a;
    h.used = a + bytes;
    return cudaSuccess;
}

inline cudaError_t hc_put(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return cudaSuccess;
    if (bytes > HC_DIRECT) return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    size_t at;
    const cudaError_t e = hc_room(bytes, &at);
    if (e != cudaSuccess) return e;
    memcpy(hc_state.stage + at, src, bytes);
    return hc_copy(dst, hc_state.dev + at, bytes);
}

// a __constant__ symbol, which no kernel may write: the one upload that stays on a copy engine
template <class T>
inline cudaError_t hc_put_symbol(const T& symbol, const void* src, size_t bytes) {
    if (bytes == 0) return cudaSuccess;
    size_t at;
    const cudaError_t e = hc_room(bytes, &at);
    if (e != cudaSuccess) return e;
    memcpy(hc_state.stage + at, src, bytes);
    return cudaMemcpyToSymbolAsync(symbol, hc_state.stage + at, bytes, 0, cudaMemcpyHostToDevice, 0);
}

inline cudaError_t hc_get(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return cudaSuccess;
    if (bytes > HC_DIRECT) return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    size_t at;
    const cudaError_t e = hc_room(bytes, &at);
    if (e != cudaSuccess) return e;
    hc_state.reads.push_back(HcRead{dst, at, bytes});
    return hc_copy(hc_state.dev + at, src, bytes);
}
