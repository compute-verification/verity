// The SASS gate's test kernels (../../test_sass_gate.py): ordinary FP32 arithmetic, and the IEEE intrinsics Pearl-C's kernels
// use, which ptxas expands with range-checked .FTZ instructions in any build. The listings beside this file are CUDA 12.9.1's
// (benchmarks/pouw/harness/jobs/toolkit.sh), for NAME plain (no flag) and fast_math (--use_fast_math; -ftz=true gives the
// same bytes here, since the kernels call the intrinsics by name):
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a [--use_fast_math] -cubin ftz_probe.cu
//   nvdisasm -c ftz_probe.cubin > NAME.nvdisasm.txt
//   cuobjdump -sass ftz_probe.cubin > NAME.cuobjdump.txt
extern "C" __global__ void rho_k(float* y, const float* a, int k) {
  int i = threadIdx.x;
  y[i] = __fsqrt_rn(__fdiv_rn(a[i], (float)((k + 7) / 8)));
}
extern "C" __global__ void rcp_k(float* y, const float* a) {
  int i = threadIdx.x;
  y[i] = __frcp_rn(a[i]);
}
extern "C" __global__ void arith_k(float* y, const float* a, const float* b) {
  int i = threadIdx.x;
  y[i] = a[i] * b[i] + (a[i] - b[i]) * 0.5f;
}
