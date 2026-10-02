// Kernels the SASS gate refuses even built without flags (../../test_sass_gate.py): each has an instruction the verifier's
// scalar model can't replay and no pinned sequence covers. __frsqrt_rn is correctly rounded but ptxas builds it from two bare
// MUFU.RSQ, which the gate has no entry for; the rest are approximate or flush subnormals. refused.nvdisasm.txt is
// `nvdisasm -c` of this file built like ftz_probe.cu (plain) with CUDA 12.9.1.
extern "C" __global__ void frsqrt_k(float* y, const float* a) {
  int i = threadIdx.x;
  y[i] = __frsqrt_rn(a[i]);
}
extern "C" __global__ void rsqrtf_k(float* y, const float* a) {
  int i = threadIdx.x;
  y[i] = rsqrtf(a[i]);
}
extern "C" __global__ void expf_k(float* y, const float* a) {
  int i = threadIdx.x;
  y[i] = __expf(a[i]);
}
extern "C" __global__ void fdividef_k(float* y, const float* a, const float* b) {
  int i = threadIdx.x;
  y[i] = __fdividef(a[i], b[i]);
}
extern "C" __global__ void sqrt_approx_k(float* y, const float* a) {
  int i = threadIdx.x;
  float r;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(a[i]));
  y[i] = r;
}
extern "C" __global__ void add_ftz_k(float* y, const float* a, const float* b) {
  int i = threadIdx.x;
  float r;
  asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a[i]), "f"(b[i]));
  y[i] = r;
}
extern "C" __global__ void min_ftz_k(float* y, const float* a, const float* b) {
  int i = threadIdx.x;
  float r;
  asm("min.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a[i]), "f"(b[i]));
  y[i] = r;
}
extern "C" __global__ void setp_ftz_k(int* y, const float* a, const float* b) {
  int i = threadIdx.x;
  int r;
  asm("{ .reg .pred p; setp.lt.ftz.f32 p, %1, %2; selp.s32 %0, 1, 0, p; }" : "=r"(r) : "f"(a[i]), "f"(b[i]));
  y[i] = r;
}
extern "C" __global__ void cvt_ftz_k(unsigned* y, const float* a) {
  int i = threadIdx.x;
  unsigned r;
  asm("cvt.rzi.ftz.u32.f32 %0, %1;" : "=r"(r) : "f"(a[i]));
  y[i] = r;
}
// Exact, but its function's one division has a constant numerator, so ptxas folds the constant into that function's copy of
// the slow path, whose body then isn't the pinned one (ftz_probe.cu's cdiv_k shares its slow path and passes).
extern "C" __global__ void cdiv1_k(float* y, const float* b) {
  int i = threadIdx.x;
  y[i] = 3.0f / b[i];
}
