// Proof-unit output commitment on the GPU, built to run inside vLLM's CUDA graphs: every launch reads its ring positions
// from device counters, so a captured graph replays correctly step after step, and nothing here synchronises with the host.
//
//   leaf(site, ci)  = SHA-256(chunk_ci || u8"vcb1" || u32be(site) || u64be(ci) || u64be(n_chunks))   (256-byte chunks, 5 compressions)
//   node(l, r)      = SHA-256(l || r), r = 0^32 for the odd trailing node of a level; the step root folds every leaf of the step
//
// A step's leaves are contiguous in the digest ring from ctr[0]; each site writes its own static range [offset, offset + n).
// The SHA-256 core is native_tree.cu's (integrations/vllm/verity_vllm/commit/committer/).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__constant__ uint32_t K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return __funnelshift_r(x, x, n); }
__device__ __forceinline__ uint32_t bswap(uint32_t x) { return __byte_perm(x, 0, 0x0123); }

struct Sha {
    uint32_t h[8];
    __device__ __forceinline__ void init() {
        h[0] = 0x6a09e667u; h[1] = 0xbb67ae85u; h[2] = 0x3c6ef372u; h[3] = 0xa54ff53au;
        h[4] = 0x510e527fu; h[5] = 0x9b05688cu; h[6] = 0x1f83d9abu; h[7] = 0x5be0cd19u;
    }
    __device__ __forceinline__ void compress(uint32_t (&m)[16]) {
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            uint32_t w;
            if (i < 16) {
                w = m[i];
            } else {
                const uint32_t w15 = m[(i + 1) & 15], w2 = m[(i + 14) & 15];
                const uint32_t s0 = rotr(w15, 7) ^ rotr(w15, 18) ^ (w15 >> 3);
                const uint32_t s1 = rotr(w2, 17) ^ rotr(w2, 19) ^ (w2 >> 10);
                w = m[i & 15] + s0 + m[(i + 9) & 15] + s1;
                m[i & 15] = w;
            }
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K256[i] + w;
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
};

constexpr int CHUNK = 256;
constexpr int REDUCE_THREADS = 256;      // one launch folds 512 nodes into one

__device__ __forceinline__ void load_digest(const uint8_t *p, uint32_t (&w)[8]) {
    const uint4 *p4 = reinterpret_cast<const uint4 *>(p);
    const uint4 a = p4[0], b = p4[1];
    w[0] = bswap(a.x); w[1] = bswap(a.y); w[2] = bswap(a.z); w[3] = bswap(a.w);
    w[4] = bswap(b.x); w[5] = bswap(b.y); w[6] = bswap(b.z); w[7] = bswap(b.w);
}

__device__ __forceinline__ void store_digest(uint8_t *p, const uint32_t (&h)[8]) {
    uint4 *p4 = reinterpret_cast<uint4 *>(p);
    p4[0] = make_uint4(bswap(h[0]), bswap(h[1]), bswap(h[2]), bswap(h[3]));
    p4[1] = make_uint4(bswap(h[4]), bswap(h[5]), bswap(h[6]), bswap(h[7]));
}

__device__ __forceinline__ void node(const uint32_t (&l)[8], const uint32_t (&r)[8], uint32_t (&out)[8]) {
    Sha s; s.init();
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = l[i]; m[8 + i] = r[i]; }
    s.compress(m);
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = 0u;
    m[0] = 0x80000000u; m[15] = 512u;
    s.compress(m);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = s.h[i];
}

// one thread per chunk; ctr[0] = the step's first digest-ring slot, ctr[1] = its first retention-ring chunk
__global__ void leaf_kernel(const uint4 *__restrict__ x, long n_chunks, uint32_t site, long offset, const int64_t *__restrict__ ctr,
                            uint8_t *__restrict__ ring, long R, uint4 *__restrict__ keep, long RH) {
    const long c0 = (long)blockIdx.x * blockDim.x;
    if (keep) {
        // Block-cooperative copy so consecutive threads store consecutive 16 B to mapped host memory.
        const long span = min((long)blockDim.x, n_chunks - c0) * (CHUNK / 16);
        const long kb = ctr[1] + offset + c0;
        for (long e = threadIdx.x; e < span; e += blockDim.x) {
            const long c = e / (CHUNK / 16);
            keep[((kb + c) % RH) * (CHUNK / 16) + e % (CHUNK / 16)] = x[(c0 + c) * (CHUNK / 16) + e % (CHUNK / 16)];
        }
    }
    const long ci = c0 + threadIdx.x;
    if (ci >= n_chunks) return;
    Sha s; s.init();
    uint32_t m[16];
    const uint4 *src = x + ci * (CHUNK / 16);
    #pragma unroll
    for (int blk = 0; blk < CHUNK / 64; ++blk) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 v = src[blk * 4 + q];
            m[4 * q + 0] = bswap(v.x); m[4 * q + 1] = bswap(v.y); m[4 * q + 2] = bswap(v.z); m[4 * q + 3] = bswap(v.w);
        }
        s.compress(m);
    }
    m[0] = 0x76636231u; m[1] = site; m[2] = (uint32_t)((unsigned long)ci >> 32); m[3] = (uint32_t)ci;
    m[4] = (uint32_t)((unsigned long)n_chunks >> 32); m[5] = (uint32_t)n_chunks; m[6] = 0x80000000u;
    #pragma unroll
    for (int i = 7; i < 15; ++i) m[i] = 0u;
    m[15] = (CHUNK + 24) * 8;
    s.compress(m);
    store_digest(ring + ((ctr[0] + offset + ci) % R) * 32, s.h);
}

// folds up to nine levels in shared memory: block b turns nodes [512b, 512b + 512) of a level of n nodes into node b of the level
// nine above, stopping early once the level has one node (the root).  Level 0 is read from the ring (wrapping from ctr[0]).
__global__ void reduce_kernel(const uint8_t *__restrict__ in, long n, const int64_t *__restrict__ ctr, long R, uint8_t *__restrict__ out) {
    __shared__ uint32_t sm[REDUCE_THREADS][8];
    const long b = blockIdx.x;
    const int t = threadIdx.x;
    const long base = ctr ? ctr[0] : 0;
    long nl = n;
    {
        const long p = b * REDUCE_THREADS + t;
        uint32_t l[8], r[8], d[8];
        if (2 * p < nl) {
            load_digest(in + (ctr ? (base + 2 * p) % R : 2 * p) * 32, l);
            if (2 * p + 1 < nl) load_digest(in + (ctr ? (base + 2 * p + 1) % R : 2 * p + 1) * 32, r);
            else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) r[i] = 0u;
            }
            node(l, r, d);
            #pragma unroll
            for (int i = 0; i < 8; ++i) sm[t][i] = d[i];
        }
        nl = (nl + 1) / 2;
    }
    for (int w = REDUCE_THREADS / 2; w >= 1; w >>= 1) {
        __syncthreads();
        if (nl == 1) break;
        const long pp = b * w + t;
        const bool have = t < w && 2 * pp < nl;
        uint32_t d[8];
        if (have) {
            uint32_t l[8], r[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) { l[i] = sm[2 * t][i]; r[i] = (2 * pp + 1 < nl) ? sm[2 * t + 1][i] : 0u; }
            node(l, r, d);
        }
        __syncthreads();
        if (have) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) sm[t][i] = d[i];
        }
        nl = (nl + 1) / 2;
    }
    __syncthreads();
    if (t == 0) {
        uint32_t d[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) d[i] = sm[0][i];
        store_digest(out + b * 32, d);
    }
}

// the step record (64 bytes, host-mapped): root[32] | u64 n_leaves | u64 ring slot | u64 retention chunk | u64 step index
__global__ void finalize_kernel(const uint8_t *__restrict__ root, long n, int64_t nbytes, int64_t *ctr, long R, long RH,
                                uint8_t *rec, long N, volatile int64_t *mirror) {
    const int64_t step = ctr[2];
    uint8_t *r = rec + (step % N) * 64;
    const uint4 *src = reinterpret_cast<const uint4 *>(root);
    reinterpret_cast<uint4 *>(r)[0] = src[0];
    reinterpret_cast<uint4 *>(r)[1] = src[1];
    int64_t *f = reinterpret_cast<int64_t *>(r + 32);
    f[0] = n; f[1] = ctr[0]; f[2] = ctr[1]; f[3] = step;
    ctr[0] = (ctr[0] + n) % R;
    ctr[1] = RH ? (ctr[1] + n) % RH : 0;
    ctr[2] = step + 1;
    ctr[3] += nbytes;
    ctr[4] += n;
    __threadfence_system();
    #pragma unroll
    for (int i = 0; i < 5; ++i) mirror[i] = ctr[i];
}

}  // namespace

// the device address of pinned host memory (resolved once, outside any capture)
int64_t device_pointer(const at::Tensor &host) {
    TORCH_CHECK(host.is_pinned(), "device_pointer: pinned host tensor");
    void *dev = nullptr;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&dev, host.data_ptr(), 0));
    return reinterpret_cast<int64_t>(dev);
}

// x: contiguous CUDA tensor of a multiple of 256 bytes; keep: device address of a pinned retention ring of RH chunks (0 = none)
void leaf_hash(const at::Tensor &x, int64_t site, int64_t offset, const at::Tensor &ctr, const at::Tensor &ring, int64_t keep, int64_t RH) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "leaf_hash: contiguous CUDA tensor");
    const long nbytes = (long)(x.numel() * x.element_size());
    TORCH_CHECK(nbytes % CHUNK == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "leaf_hash: 256-byte multiple, 16-byte aligned");
    const long n = nbytes / CHUNK;
    if (n == 0) return;
    const int threads = 64;
    leaf_kernel<<<(n + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4 *>(x.data_ptr()), n, (uint32_t)site, (long)offset, ctr.data_ptr<int64_t>(),
        ring.data_ptr<uint8_t>(), ring.numel() / 32, reinterpret_cast<uint4 *>(keep), RH);
    AT_CUDA_CHECK(cudaGetLastError());
}

// folds the step's n leaves to its root, writes the step record (rec: device address of N pinned 64-byte records) and advances
// the counters (mirror: device address of 5 pinned int64); scratch: [2, cap, 32] uint8
void finalize(int64_t n, int64_t nbytes, const at::Tensor &ctr, const at::Tensor &ring, const at::Tensor &scratch, int64_t RH,
              int64_t rec, int64_t N, int64_t mirror) {
    TORCH_CHECK(n >= 2, "finalize: a step has at least two leaves");
    const long R = ring.numel() / 32, cap = scratch.size(1);
    TORCH_CHECK(n <= R, "finalize: the step's leaves exceed the digest ring");
    auto st = at::cuda::getCurrentCUDAStream();
    uint8_t *buf[2] = {scratch[0].data_ptr<uint8_t>(), scratch[1].data_ptr<uint8_t>()};
    const uint8_t *root = nullptr;
    long cur = n;
    int which = 0;
    const int64_t *c = ctr.data_ptr<int64_t>();
    const uint8_t *in = ring.data_ptr<uint8_t>();
    bool from_ring = true;
    while (cur > 1) {
        const long blocks = (cur + 2 * REDUCE_THREADS - 1) / (2 * REDUCE_THREADS);
        TORCH_CHECK(blocks <= cap, "finalize: tree scratch too small");
        reduce_kernel<<<blocks, REDUCE_THREADS, 0, st>>>(in, cur, from_ring ? c : nullptr, R, buf[which]);
        AT_CUDA_CHECK(cudaGetLastError());
        in = buf[which]; root = buf[which]; from_ring = false;
        which ^= 1;
        cur = blocks;
    }
    finalize_kernel<<<1, 1, 0, st>>>(root, n, nbytes, ctr.data_ptr<int64_t>(), R, RH, reinterpret_cast<uint8_t *>(rec), N,
                                     reinterpret_cast<int64_t *>(mirror));
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("device_pointer", &device_pointer, "device address of pinned host memory");
    m.def("leaf_hash", &leaf_hash, "SHA-256 leaves of x's 256-byte chunks into the digest ring at the step's offset");
    m.def("finalize", &finalize, "fold the step's leaves to its root, record it host-side and advance the counters");
}
