// Does a buffer the size of a layer's sketches, written before S bytes of streamed weight reads, still hit in L2 when
// it is read back (lane ra-rate-redteam)? ra-sketch-tag's kernel writes every tile's sketch to global memory and a
// later call's committer warps read it, with the weights' loads at L2's default priority in between; a read-back that
// misses L2 is HBM ingress the audit doesn't certify.
//
//   l2keep B S HINT REPS          sizes in MiB. Each rep: flush (stream 512 MiB elsewhere), fill B, stream S, read B back; prints each
//                                 kernel's time. HINT 0 default; 1 the stream at L2::evict_first; 2 the fill at
//                                 L2::evict_last. Under ncu, profile the readback kernel for its DRAM bytes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __forceinline__ uint64_t policy_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__device__ __forceinline__ uint64_t policy_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__global__ void fill(uint4* b, size_t n, uint32_t v, int last) {
  const uint64_t pol = last ? policy_last() : 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 x = make_uint4((uint32_t)i ^ v, (uint32_t)i + v, (uint32_t)i * 3u, v);
    if (last)
      asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(b + i), "r"(x.x), "r"(x.y),
                   "r"(x.z), "r"(x.w), "l"(pol)
                   : "memory");
    else
      b[i] = x;
  }
}

__global__ void stream(const uint4* w, size_t n, int first, unsigned* out) {
  const uint64_t pol = first ? policy_first() : 0;
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 x;
    if (first)
      asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                   : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
                   : "l"(w + i), "l"(pol));
    else
      x = w[i];
    acc ^= x.x ^ x.y ^ x.z ^ x.w;
  }
  if (acc == 0x9E3779B9u) out[0] = acc;
}

__global__ void readback(const uint4* b, size_t n, unsigned* out) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 x = b[i];
    acc ^= x.x ^ x.y ^ x.z ^ x.w;
  }
  if (acc == 0x9E3779B9u) out[0] = acc;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: l2keep B_MB S_MB HINT REPS\n");
    return 64;
  }
  const size_t bmb = atoll(argv[1]), smb = atoll(argv[2]);
  const int hint = atoi(argv[3]), reps = atoi(argv[4]);
  const size_t nb = bmb << 16, ns = smb << 16, nf = (size_t)512 << 16;  // uint4 counts
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  uint4 *b, *w = nullptr, *f;
  unsigned* out;
  if (cudaMalloc(&b, nb * 16) || cudaMalloc(&f, nf * 16) || cudaMalloc(&out, 4) || (ns && cudaMalloc(&w, ns * 16)))
    return 65;
  cudaMemset(f, 0x5A, nf * 16);
  if (ns) cudaMemset(w, 0xA5, ns * 16);
  cudaEvent_t e[4];
  for (auto& x : e) cudaEventCreate(&x);
  const int grid = sms * 8, threads = 256;
  for (int r = 0; r < reps; ++r) {
    stream<<<grid, threads>>>(f, nf, 0, out);
    cudaEventRecord(e[0]);
    fill<<<grid, threads>>>(b, nb, 0x1234u + r, hint == 2);
    cudaEventRecord(e[1]);
    if (ns) stream<<<grid, threads>>>(w, ns, hint == 1, out);
    cudaEventRecord(e[2]);
    readback<<<grid, threads>>>(b, nb, out);
    cudaEventRecord(e[3]);
    cudaEventSynchronize(e[3]);
    float t[3];
    for (int i = 0; i < 3; ++i) cudaEventElapsedTime(&t[i], e[i], e[i + 1]);
    printf("B %zu MiB S %zu MiB hint %d rep %d: fill %.4f ms, stream %.4f ms, readback %.4f ms (%.1f GB/s)\n", bmb, smb,
           hint, r, t[0], t[1], t[2], nb * 16 / (t[2] * 1e6));
  }
  return cudaGetLastError() ? 3 : 0;
}
