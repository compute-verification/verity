// Read kernels on the RTX PRO 6000 (sm_120): which reader gets closest to the card's read capacity, and so what R is.
// η can be no lower than the best reader's shortfall from R, so R is the best of every reader here, measured in the
// same run (`fill.py read`).
//
//   ld<U>     grid-stride, each thread U independent 16-byte loads in flight (ld.global.nc.L1::no_allocate), XOR-folded
//   ldef<U>   the same with an L2 evict-first policy
//   bulk<S>   cp.async.bulk (TMA) into S stages of BULK_BYTES of shared memory per CTA, one elected thread issuing,
//             every thread folding each stage once it lands
//
// Built as libreaders.so (fill_run.sh). C ABI at the end.
#include <cuda_runtime.h>
#include <stdint.h>

namespace rd {

__device__ __forceinline__ uint4 ld_na(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ uint4 ld_ef(const uint4* p, uint64_t pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}

template <int U, bool EF>
__global__ void __launch_bounds__(256) ld_kernel(const uint4* __restrict__ p, uint64_t n16, uint32_t* out) {
  uint64_t pol = 0;
  if (EF) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  uint32_t acc = 0;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n16; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = EF ? ld_ef(p + i + u * stride, pol) : ld_na(p + i + u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n16; i += stride) {
    const uint4 v = ld_na(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) atomicXor(out, acc);
}

constexpr uint32_t BULK_BYTES = 16384;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
  asm volatile(
      "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n" ::"r"(smem_u32(b)),
      "r"(phase)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}

// CTA b takes BULK_BYTES pieces b, b + grid, ...; S stages in flight.
template <int S>
__global__ void __launch_bounds__(128) bulk_kernel(const uint8_t* __restrict__ p, uint64_t pieces, uint32_t* out) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ uint64_t bar[S];
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  if (threadIdx.x == 0)
    for (int s = 0; s < S; ++s) mbar_init(bar + s, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  const uint64_t first = blockIdx.x, step = gridDim.x;
  if (threadIdx.x == 0)
    for (int s = 0; s < S; ++s) {
      const uint64_t q = first + s * step;
      if (q < pieces) {
        mbar_expect(bar + s, BULK_BYTES);
        bulk_g2s(sm + s * BULK_BYTES, p + q * BULK_BYTES, BULK_BYTES, bar + s, pol);
      }
    }
  uint32_t acc = 0;
  uint32_t k = 0;
  for (uint64_t q = first; q < pieces; q += step, ++k) {
    const int s = k % S;
    mbar_wait(bar + s, (k / S) & 1);
    const uint4* v = (const uint4*)(sm + s * BULK_BYTES);
#pragma unroll 4
    for (uint32_t i = threadIdx.x; i < BULK_BYTES / 16; i += 128) {
      const uint4 w = v[i];
      acc ^= w.x ^ w.y ^ w.z ^ w.w;
    }
    __syncthreads();
    const uint64_t nq = q + (uint64_t)S * step;
    if (threadIdx.x == 0 && nq < pieces) {
      mbar_expect(bar + s, BULK_BYTES);
      bulk_g2s(sm + s * BULK_BYTES, p + nq * BULK_BYTES, BULK_BYTES, bar + s, pol);
    }
  }
  if (acc == 0x9e3779b9u) atomicXor(out, acc);
}

template <int S>
static int launch_bulk(const void* p, uint64_t bytes, int grid, cudaStream_t st, uint32_t* xr) {
  const size_t smem = (size_t)S * BULK_BYTES;
  cudaFuncSetAttribute(bulk_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  bulk_kernel<S><<<grid, 128, smem, st>>>((const uint8_t*)p, bytes / BULK_BYTES, xr);
  return (int)cudaGetLastError();
}

}  // namespace rd

extern "C" {

// Reader `kind` over `bytes` at `p` with `grid` CTAs: 0 ld<U>, 1 ldef<U>, 2 bulk<U>; U in {1, 2, 4, 8} (bulk: 2, 4, 6).
int rd_read(int kind, int u, const void* p, uint64_t bytes, int grid, void* stream, uint32_t* xr) {
  cudaStream_t st = (cudaStream_t)stream;
  const uint64_t n16 = bytes / 16;
  using namespace rd;
#define LD(U, EF) ld_kernel<U, EF><<<grid, 256, 0, st>>>((const uint4*)p, n16, xr)
  if (kind == 0 || kind == 1) {
    const bool ef = kind == 1;
    switch (u) {
      case 1: ef ? LD(1, true) : LD(1, false); break;
      case 2: ef ? LD(2, true) : LD(2, false); break;
      case 4: ef ? LD(4, true) : LD(4, false); break;
      case 8: ef ? LD(8, true) : LD(8, false); break;
      default: return -1;
    }
    return (int)cudaGetLastError();
  }
#undef LD
  if (kind == 2) {
    switch (u) {
      case 2: return launch_bulk<2>(p, bytes, grid, st, xr);
      case 4: return launch_bulk<4>(p, bytes, grid, st, xr);
      case 6: return launch_bulk<6>(p, bytes, grid, st, xr);
      default: return -1;
    }
  }
  return -1;
}

int rd_sms() {
  int d = 0, n = 0;
  cudaGetDevice(&d);
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
  return n;
}

}  // extern "C"
