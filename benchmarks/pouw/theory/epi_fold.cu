// The int8 GEMM's epilogue, one output row a block: s32 accumulator -> f32 -> per-row and per-column scales -> bf16.
// MODE 0 stock (I2F, x * s_a, * s_w); 1 fold as an FFMA bias (x * s_a + c_i, c_i = -r_i s_a a row);
// 2 fold exact in the integer domain (acc - r_i, then as stock); 3 stock by the magic-number conversion
// (int_as_float(acc + 0x4B400000) - 1.5 * 2^23, exact for |acc| < 2^22); 4 its fold (the magic less r_i a row).
#include <cuda_bf16.h>

template <int MODE>
__global__ void epi(const int* __restrict__ acc, const float* __restrict__ sa, const float* __restrict__ sw,
                    const int* __restrict__ rowterm, __nv_bfloat162* __restrict__ out, int n) {
  const int row = blockIdx.x;
  const float a = sa[row];
  const int r = (MODE == 1 || MODE == 2 || MODE == 4) ? rowterm[row] : 0;
  const float c = -(float)r * a;
  const int magic = 0x4B400000 - (MODE == 4 ? r : 0);
  const int* arow = acc + (size_t)row * n;
  __nv_bfloat162* orow = out + (size_t)row * n / 2;
#pragma unroll 8
  for (int j = 2 * threadIdx.x; j < n; j += 2 * blockDim.x) {
    int2 v = *reinterpret_cast<const int2*>(arow + j);
    float2 w = *reinterpret_cast<const float2*>(sw + j);
    float x0, x1;
    if (MODE == 3 || MODE == 4) {
      x0 = __int_as_float(v.x + magic) - 12582912.0f;
      x1 = __int_as_float(v.y + magic) - 12582912.0f;
    } else if (MODE == 2) {
      x0 = (float)(v.x - r);
      x1 = (float)(v.y - r);
    } else {
      x0 = (float)v.x;
      x1 = (float)v.y;
    }
    float y0 = MODE == 1 ? fmaf(x0, a, c) : x0 * a;
    float y1 = MODE == 1 ? fmaf(x1, a, c) : x1 * a;
    orow[j / 2] = __floats2bfloat162_rn(y0 * w.x, y1 * w.y);
  }
}

template __global__ void epi<0>(const int*, const float*, const float*, const int*, __nv_bfloat162*, int);
template __global__ void epi<1>(const int*, const float*, const float*, const int*, __nv_bfloat162*, int);
template __global__ void epi<2>(const int*, const float*, const float*, const int*, __nv_bfloat162*, int);
template __global__ void epi<3>(const int*, const float*, const float*, const int*, __nv_bfloat162*, int);
template __global__ void epi<4>(const int*, const float*, const float*, const int*, __nv_bfloat162*, int);
