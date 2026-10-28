// Device-side weight registration: the SAME leaves and the SAME tree as native_host.register_weights' host path
// (verity_vllm.commit.scheme.pos_leaf over CHUNK-byte position-identified leaves, scheme.fold: node / lift rule), computed on the
// GPU where the parameters already live.  The host path hashes 256-byte leaves in Python (~0.9 us per leaf + the fold): 2.5 GB of weights =
// 10 M leaves = 17-24 s per registration, 120 s for a 7B model, and a Commit registers twice (before the workload, again at the C2 replay
// for the O5 weights check).  Same bytes in -> same root out; equality with the host path is asserted by the tests on many sizes (tail
// leaves shorter than a chunk included) and, in `device-checked` mode, at runtime on the real model.
//
//   pos_leaf(value)  = SHA-256("verity/pos-leaf/v0" || u64be(len(value)) || value)          (scheme.pos_leaf)
//   node(l, r)       = SHA-256(NODE_PREFIX || l || u64be(32) || r), lift(c) = SHA-256(LIFT_PREFIX || c)   (scheme.fold: odd trailing node)
//
// SHA-256 core and tree_level_kernel are VERBATIM native_tree.cu (checked against hashlib there); only pos_leaves_kernel is new.
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

struct Blk {
    uint32_t w[16];
    __device__ __forceinline__ void clear() {
        #pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = 0u;
    }
    __device__ __forceinline__ void byte(int i, uint32_t b) { w[i >> 2] |= (b & 0xFFu) << (24 - 8 * (i & 3)); }
};

__constant__ uint8_t NODE_PREFIX[31] = {0, 0, 0, 0x13, 'v', 'e', 'r', 'i', 't', 'y', '-', 'v', 'l', 'l', 'm', '/', 'n', 'o', 'd', 'e', '/', 'v', '1', 0, 0, 0, 0, 0, 0, 0, 0x20};
__constant__ uint8_t LIFT_PREFIX[31] = {0, 0, 0, 0x13, 'v', 'e', 'r', 'i', 't', 'y', '-', 'v', 'l', 'l', 'm', '/', 'l', 'i', 'f', 't', '/', 'v', '1', 0, 0, 0, 0, 0, 0, 0, 0x20};
__constant__ uint8_t POS_LEAF_TAG[18] = {'v', 'e', 'r', 'i', 't', 'y', '/', 'p', 'o', 's', '-', 'l', 'e', 'a', 'f', '/', 'v', '0'};

__device__ __forceinline__ uint32_t dbyte(const uint32_t *d, int j) { return (d[j >> 2] >> (8 * (j & 3))) & 0xFFu; }

// leaf ci = SHA-256(POS_LEAF_TAG[18] || u64be(len) || buf[ci*chunk : ci*chunk+len]), len = min(chunk, nbytes - ci*chunk) (the last leaf may
// be shorter -- exactly the host path's mv[o:o+chunk]).  Byte-wise message assembly (one thread per leaf; the message is 26 + len bytes).
__global__ void pos_leaves_kernel(const uint8_t *__restrict__ buf, long nbytes, int chunk, uint32_t *__restrict__ out, long n_leaves) {
    const long ci = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_leaves) return;
    const long off = ci * (long)chunk;
    const long len = (nbytes - off) < chunk ? (nbytes - off) : chunk;
    const long msg = 26 + len;                       // bytes before padding
    const long total = ((msg + 8) / 64 + 1) * 64;    // padded length (0x80, zeros, u64be bit length)
    const uint8_t *src = buf + off;
    Sha s; s.init();
    Blk b;
    for (long base = 0; base < total; base += 64) {
        b.clear();
        for (int i = 0; i < 64; ++i) {
            const long p = base + i;
            uint32_t v;
            if (p < 18) v = POS_LEAF_TAG[p];
            else if (p < 26) v = (uint32_t)((((uint64_t)len) >> (8 * (25 - p))) & 0xFFu);
            else if (p < msg) v = src[p - 26];
            else if (p == msg) v = 0x80u;
            else if (p >= total - 8) v = (uint32_t)((((uint64_t)msg * 8u) >> (8 * (total - 1 - p))) & 0xFFu);
            else v = 0u;
            b.byte(i, v);
        }
        s.compress(b.w);
    }
    uint4 *o4 = reinterpret_cast<uint4 *>(out + ci * 8);
    o4[0] = make_uint4(bswap(s.h[0]), bswap(s.h[1]), bswap(s.h[2]), bswap(s.h[3]));
    o4[1] = make_uint4(bswap(s.h[4]), bswap(s.h[5]), bswap(s.h[6]), bswap(s.h[7]));
}

// one level: parent i = node(child 2i, child 2i+1) or lift(child 2i) when 2i is the odd trailing node  (VERBATIM native_tree.cu)
__global__ void tree_level_kernel(const uint32_t *__restrict__ in, uint32_t *__restrict__ out, long n_in) {
    const long pi = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long n_out = (n_in + 1) / 2;
    if (pi >= n_out) return;
    const uint32_t *L = in + 2 * pi * 8;
    const bool lift = (2 * pi + 1 >= n_in);
    Sha s; s.init();
    Blk b; b.clear();
    if (!lift) {
        const uint32_t *R = L + 8;
        #pragma unroll
        for (int i = 0; i < 31; ++i) b.byte(i, NODE_PREFIX[i]);
        #pragma unroll
        for (int j = 0; j < 32; ++j) b.byte(31 + j, dbyte(L, j));
        s.compress(b.w);
        b.clear();
        b.byte(6, 0x20u);
        #pragma unroll
        for (int j = 0; j < 32; ++j) b.byte(7 + j, dbyte(R, j));
        b.byte(39, 0x80u);
        b.w[15] = 824u;
        s.compress(b.w);
    } else {
        #pragma unroll
        for (int i = 0; i < 31; ++i) b.byte(i, LIFT_PREFIX[i]);
        #pragma unroll
        for (int j = 0; j < 32; ++j) b.byte(31 + j, dbyte(L, j));
        b.byte(63, 0x80u);
        s.compress(b.w);
        b.clear();
        b.w[15] = 504u;
        s.compress(b.w);
    }
    uint4 *o4 = reinterpret_cast<uint4 *>(out + pi * 8);
    o4[0] = make_uint4(bswap(s.h[0]), bswap(s.h[1]), bswap(s.h[2]), bswap(s.h[3]));
    o4[1] = make_uint4(bswap(s.h[4]), bswap(s.h[5]), bswap(s.h[6]), bswap(s.h[7]));
}

}  // namespace

// buf: contiguous CUDA uint8 tensor (any length >= 1) -> [n_leaves, 32] uint8 CUDA leaf digests, tree order
at::Tensor pos_leaves(const at::Tensor &buf, int64_t chunk) {
    TORCH_CHECK(buf.is_cuda() && buf.scalar_type() == at::kByte && buf.is_contiguous(), "pos_leaves: contiguous CUDA uint8 tensor required");
    TORCH_CHECK(chunk > 0, "pos_leaves: chunk > 0");
    const long nbytes = (long)buf.numel();
    TORCH_CHECK(nbytes > 0, "pos_leaves: empty buffer has no leaves (the host path skips numel()==0 tensors)");
    const long n_leaves = (nbytes + chunk - 1) / chunk;
    auto out = at::empty({n_leaves, 32}, buf.options());
    const int threads = 128;
    const long blocks = (n_leaves + threads - 1) / threads;
    auto st = at::cuda::getCurrentCUDAStream();
    pos_leaves_kernel<<<blocks, threads, 0, st.stream()>>>(buf.data_ptr<uint8_t>(), nbytes, (int)chunk,
                                                            reinterpret_cast<uint32_t *>(out.data_ptr<uint8_t>()), n_leaves);
    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}

// leaves: [n, 32] uint8 CUDA -> [32] uint8 CUDA root (scheme.fold: node / lift levels down to one node)
at::Tensor tree_root(const at::Tensor &leaves) {
    TORCH_CHECK(leaves.is_cuda() && leaves.scalar_type() == at::kByte && leaves.is_contiguous() && leaves.dim() == 2 && leaves.size(1) == 32,
                "tree_root: [n, 32] contiguous CUDA uint8 leaves required");
    long n = (long)leaves.size(0);
    TORCH_CHECK(n > 0, "tree_root: no leaves");
    auto st = at::cuda::getCurrentCUDAStream();
    at::Tensor cur = leaves;
    while (n > 1) {
        const long n_out = (n + 1) / 2;
        auto nxt = at::empty({n_out, 32}, leaves.options());
        const int threads = 128;
        const long blocks = (n_out + threads - 1) / threads;
        tree_level_kernel<<<blocks, threads, 0, st.stream()>>>(reinterpret_cast<const uint32_t *>(cur.data_ptr<uint8_t>()),
                                                                reinterpret_cast<uint32_t *>(nxt.data_ptr<uint8_t>()), n);
        AT_CUDA_CHECK(cudaGetLastError());
        cur = nxt;
        n = n_out;
    }
    return cur.reshape({32});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pos_leaves", &pos_leaves, "pos_leaf digests of CHUNK-byte leaves of a CUDA uint8 buffer -> [n, 32] uint8");
    m.def("tree_root", &tree_root, "scheme.fold of [n, 32] leaf digests -> [32] uint8 root");
}
