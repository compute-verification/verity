// One kernel with no floating-point division, square root or reciprocal: nothing the gate flags.
extern "C" __global__ void scale_add(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 2.0f + 1.0f;
}
