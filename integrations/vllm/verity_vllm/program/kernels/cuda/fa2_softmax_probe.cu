// The FA2 online-softmax scalar path exactly as nvcc compiles it under --use_fast_math:
//   p = exp2f(s * scale - max_scaled)         (FFMA.FTZ + MUFU.EX2)
//   sc = exp2f((m_prev - m_cur) * scale)      (FADD.FTZ, FMUL.FTZ, MUFU.EX2)
//   inv = 1.f / sum                            (MUFU.RCP)
//   per-lane row sum of a 64-key block as the production kernel is COMPILED: nvcc contracts
//   `row_sum *= scores_scale; row_sum += p[first col]` into FFMA.FTZ(scores_scale, l, p0)
//   (seen in the production SASS), followed by 15 sequential FADD.FTZ in the lane's element order.
//   Written explicitly with fmaf here because a conditional multiply is not contracted by nvcc.
// GPU host only.  Build: nvcc -O2 --use_fast_math -arch=sm_89 -Xcompiler -fPIC -shared -o libfa2_softmax_probe.so fa2_softmax_probe.cu
#include <cstdint>
#include <cuda_runtime.h>

__global__ void k_p(const float* s, float scale, float max_scaled, float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = exp2f(s[i] * scale - max_scaled);
}
__global__ void k_scale(const float* mprev, const float* mcur, float scale, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp2f((mprev[i] - mcur[i]) * scale);
}
__global__ void k_rcp(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float sum = x[i]; y[i] = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum; }
}
__global__ void k_lanesum(const float* p, const float* l_in, const float* sc, int rescale, float* l_out) {
    int qd = threadIdx.x;
    if (qd >= 4) return;
    float l = l_in[qd];
    l = rescale ? fmaf(sc[0], l, p[2 * qd]) : l + p[2 * qd];
#pragma unroll
    for (int nj = 0; nj < 8; ++nj)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            if (nj || j) l = l + p[8 * nj + 2 * qd + j];
    l_out[qd] = l;
}
__global__ void k_maxscaled(const float* m, float scale, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = m[i] == -INFINITY ? 0.f : m[i] * scale;
}

extern "C" {
int fa2_p(const float* s, float scale, float max_scaled, float* p, int n) {
    float *ds, *dp;
    cudaMalloc(&ds, n * 4); cudaMalloc(&dp, n * 4);
    cudaMemcpy(ds, s, n * 4, cudaMemcpyHostToDevice);
    k_p<<<(n + 255) / 256, 256>>>(ds, scale, max_scaled, dp, n);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(p, dp, n * 4, cudaMemcpyDeviceToHost);
    cudaFree(ds); cudaFree(dp);
    return (int)e;
}
int fa2_scale(const float* mprev, const float* mcur, float scale, float* out, int n) {
    float *a, *b, *c;
    cudaMalloc(&a, n * 4); cudaMalloc(&b, n * 4); cudaMalloc(&c, n * 4);
    cudaMemcpy(a, mprev, n * 4, cudaMemcpyHostToDevice); cudaMemcpy(b, mcur, n * 4, cudaMemcpyHostToDevice);
    k_scale<<<(n + 255) / 256, 256>>>(a, b, scale, c, n);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(out, c, n * 4, cudaMemcpyDeviceToHost);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return (int)e;
}
int fa2_rcp(const float* x, float* y, int n) {
    float *a, *b;
    cudaMalloc(&a, n * 4); cudaMalloc(&b, n * 4);
    cudaMemcpy(a, x, n * 4, cudaMemcpyHostToDevice);
    k_rcp<<<(n + 255) / 256, 256>>>(a, b, n);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(y, b, n * 4, cudaMemcpyDeviceToHost);
    cudaFree(a); cudaFree(b);
    return (int)e;
}
int fa2_lanesum(const float* p, const float* l_in, float sc, int rescale, float* l_out) {
    float *dp, *dl, *dsc, *dout;
    cudaMalloc(&dp, 64 * 4); cudaMalloc(&dl, 4 * 4); cudaMalloc(&dsc, 4); cudaMalloc(&dout, 4 * 4);
    cudaMemcpy(dp, p, 64 * 4, cudaMemcpyHostToDevice); cudaMemcpy(dl, l_in, 4 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsc, &sc, 4, cudaMemcpyHostToDevice);
    k_lanesum<<<1, 4>>>(dp, dl, dsc, rescale, dout);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(l_out, dout, 4 * 4, cudaMemcpyDeviceToHost);
    cudaFree(dp); cudaFree(dl); cudaFree(dsc); cudaFree(dout);
    return (int)e;
}
int fa2_maxscaled(const float* m, float scale, float* y, int n) {
    float *a, *b;
    cudaMalloc(&a, n * 4); cudaMalloc(&b, n * 4);
    cudaMemcpy(a, m, n * 4, cudaMemcpyHostToDevice);
    k_maxscaled<<<(n + 255) / 256, 256>>>(a, scale, b, n);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(y, b, n * 4, cudaMemcpyDeviceToHost);
    cudaFree(a); cudaFree(b);
    return (int)e;
}
}
