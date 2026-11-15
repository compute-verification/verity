// The SASS gate's test kernels (../../test_sass_gate.py): ordinary FP32 arithmetic, and every sequence the gate pins
// (harness/sass_pins.json). ptxas expands div.rn (also with a constant numerator), sqrt.rn and rcp.rn with .FTZ instructions
// and MUFU seeds in every build, and seeds integer division (32-, 16- and 64-bit, and a uniform divisor) with MUFU.RCP and
// F2I.FTZ. Built without flags, every kernel passes; built with --use_fast_math, -ftz=true or -prec-div=false
// -prec-sqrt=false, the FP32 kernels fail. The listings beside this file: NAME.nvdisasm.txt from `nvdisasm -c` of
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a FLAGS -cubin ftz_probe.cu
// with CUDA 12.9.1 (harness/jobs/toolkit.sh) for NAME plain (no FLAGS), fast_math, ftz and prec, and with CUDA 13.0.1 for
// plain13; plain.cuobjdump.txt and fast_math.cuobjdump.txt from `cuobjdump -sass` of those cubins, their encodings dropped;
// and tool_notes.json, each cubin's .note.nv.tkinfo (sass_gate.tool_notes).
extern "C" __global__ void rho_k(float* y, const float* a, int k) {
  int i = threadIdx.x;
  y[i] = __fsqrt_rn(__fdiv_rn(a[i], (float)((k + 7) / 8)));
}
extern "C" __global__ void rcp_k(float* y, const float* a) {
  int i = threadIdx.x;
  y[i] = __frcp_rn(a[i]);
}
extern "C" __global__ void div_k(float* y, const float* a, const float* b) {
  int i = threadIdx.x;
  y[i] = a[i] / b[i];
}
extern "C" __global__ void cdiv_k(float* y, const float* a, const float* b) {
  int i = threadIdx.x;
  y[i] = 3.0f / b[i] - a[i] / b[i];
}
extern "C" __global__ void sqrt_k(float* y, const float* a) {
  int i = threadIdx.x;
  y[i] = sqrtf(a[i]);
}
extern "C" __global__ void arith_k(float* y, const float* a, const float* b) {
  int i = threadIdx.x;
  y[i] = a[i] * b[i] + (a[i] - b[i]) * 0.5f;
}
extern "C" __global__ void udiv_k(unsigned* y, const unsigned* a, const unsigned* b) {
  int i = threadIdx.x;
  y[i] = a[i] / b[i];
}
extern "C" __global__ void srem_k(int* y, const int* a, const int* b) {
  int i = threadIdx.x;
  y[i] = a[i] % b[i];
}
extern "C" __global__ void udivn_k(unsigned* y, const unsigned* a, unsigned n) {
  int i = threadIdx.x;
  y[i] = a[i] / n;
}
extern "C" __global__ void udiv16_k(unsigned short* y, const unsigned short* a, const unsigned short* b) {
  int i = threadIdx.x;
  unsigned short r;
  asm("div.u16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  y[i] = r;
}
extern "C" __global__ void udiv64_k(unsigned long long* y, const unsigned long long* a, const unsigned long long* b) {
  int i = threadIdx.x;
  y[i] = a[i] / b[i];
}
extern "C" __global__ void srem64_k(long long* y, const long long* a, const long long* b) {
  int i = threadIdx.x;
  y[i] = a[i] % b[i];
}
