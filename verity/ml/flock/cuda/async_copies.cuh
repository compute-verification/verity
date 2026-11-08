// verity flock: copies between the host and the default stream that hold the host only where it reads a value.
// Installed into cuda-ghash by verity/ml/flock/cuda_chunk_patch.py; the provers' headers include it.
//
// hc_put queues an upload: its bytes go into page-locked staging at once (the caller's buffer may change or go as soon as it
// returns) and the device copies them in stream order, so the host doesn't wait for the kernels queued before it. hc_get
// queues a read into staging; hc_wait synchronizes the stream once and moves every queued read into its destination, which
// must stay valid until then. A transcript read is its hc_gets then one hc_wait; a read the host doesn't need yet (an
// opening's rows and paths) waits in the queue for the next transcript read. Staging is reused only after an hc_wait, which
// completes every copy queued before it: a copy that doesn't fit waits first, and one larger than HC_DIRECT goes through
// cudaMemcpy. The state is the calling thread's.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

static constexpr size_t HC_STAGE = (size_t)16 << 20;   // staging bytes, allocated on a thread's first copy
static constexpr size_t HC_DIRECT = (size_t)4 << 20;   // copies past this many bytes are cudaMemcpy

struct HcRead { void* dst; size_t at, bytes; };
struct HostCopies {
    uint8_t* stage = nullptr;
    size_t used = 0;
    std::vector<HcRead> reads;
};
static thread_local HostCopies hc_state;

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
    if (!h.stage && cudaHostAlloc((void**)&h.stage, HC_STAGE, cudaHostAllocPortable) != cudaSuccess) {
        h.stage = nullptr;
        return cudaErrorMemoryAllocation;
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
    return cudaMemcpyAsync(dst, hc_state.stage + at, bytes, cudaMemcpyHostToDevice, 0);
}

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
    return cudaMemcpyAsync(hc_state.stage + at, src, bytes, cudaMemcpyDeviceToHost, 0);
}
