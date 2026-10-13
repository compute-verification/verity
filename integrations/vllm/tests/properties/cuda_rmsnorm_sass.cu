// Standalone instantiations of vLLM d9105ea80 layernorm_kernels.cu's served BF16 width-8 arithmetic.
// Compile exactly as the capture does:
//   nvcc -O3 -std=c++17 -lineinfo -gencode arch=compute_90,code=sm_90 -cubin cuda_rmsnorm_sass.cu -o rmsnorm.cubin
#include <cuda_bf16.h>
#include <cub/block/block_reduce.cuh>

struct alignas(16) Bf16x8 {
  __nv_bfloat16 v[8];
};

extern "C" __global__ void rms_norm_bf16_v8(
    Bf16x8* __restrict__ out, const Bf16x8* __restrict__ input,
    const Bf16x8* __restrict__ weight, float epsilon, int hidden_size) {
  __shared__ float scale;
  float variance = 0.0f;
  for (int i = threadIdx.x; i < hidden_size / 8; i += blockDim.x) {
    Bf16x8 x = input[blockIdx.x * (hidden_size / 8) + i];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float z = __bfloat162float(x.v[j]);
      variance += z * z;
    }
  }
  using Reduce = cub::BlockReduce<float, 1024>;
  __shared__ typename Reduce::TempStorage storage;
  variance = Reduce(storage).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) scale = rsqrtf(variance / hidden_size + epsilon);
  __syncthreads();
  for (int i = threadIdx.x; i < hidden_size / 8; i += blockDim.x) {
    Bf16x8 x = input[blockIdx.x * (hidden_size / 8) + i], w = weight[i], y;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      y.v[j] = __float2bfloat16(__bfloat162float(x.v[j]) * scale * __bfloat162float(w.v[j]));
    out[blockIdx.x * (hidden_size / 8) + i] = y;
  }
}

extern "C" __global__ void fused_add_rms_norm_bf16_v8(
    Bf16x8* __restrict__ input, Bf16x8* __restrict__ residual,
    const Bf16x8* __restrict__ weight, float epsilon, int hidden_size) {
  __shared__ float scale;
  float variance = 0.0f;
  for (int i = threadIdx.x; i < hidden_size / 8; i += blockDim.x) {
    int k = blockIdx.x * (hidden_size / 8) + i;
    Bf16x8 x = input[k], r = residual[k];
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      __nv_bfloat162 p = __hadd2(*reinterpret_cast<__nv_bfloat162*>(&x.v[j]),
                                 *reinterpret_cast<__nv_bfloat162*>(&r.v[j]));
      *reinterpret_cast<__nv_bfloat162*>(&x.v[j]) = p;
      float2 z = __bfloat1622float2(p);
      variance += z.x * z.x + z.y * z.y;
    }
    residual[k] = x;
  }
  using Reduce = cub::BlockReduce<float, 1024>;
  __shared__ typename Reduce::TempStorage storage;
  variance = Reduce(storage).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) scale = rsqrtf(variance / hidden_size + epsilon);
  __syncthreads();
  for (int i = threadIdx.x; i < hidden_size / 8; i += blockDim.x) {
    int k = blockIdx.x * (hidden_size / 8) + i;
    Bf16x8 x = residual[k], w = weight[i], y;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      y.v[j] = __float2bfloat16(__bfloat162float(x.v[j]) * scale * __bfloat162float(w.v[j]));
    input[k] = y;
  }
}
