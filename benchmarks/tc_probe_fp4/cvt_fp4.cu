// The FP4-path conversions on sm_120 (cast.py): one cvt per input pair or code, raw bits in and out.
//
//   op 0  cvt.rn.satfinite.e2m1x2.f32   : 8 f32 -> one u32, byte j = cvt(a = in[8t+2j], b = in[8t+2j+1]) (a -> upper nibble)
//   op 1  cvt.rn.f16x2.e2m1x2           : one e2m1x2 byte (in[t] & 0xFF) -> one f16x2 word
//   op 2  cvt.rz.satfinite.ue8m0x2.f32  : f32 pair (a = in[2t], b = in[2t+1]) -> u16 (a -> upper byte), zero-extended
//   op 3  cvt.rp.satfinite.ue8m0x2.f32  : likewise, rounding up
//   op 4  cvt.rn.bf16x2.ue8m0x2         : one ue8m0x2 half-word (in[t] & 0xFFFF) -> one bf16x2 word
//   op 5  cvt.rn.satfinite.e4m3x2.f32   : f32 pair -> u16 (a -> upper byte); the NVFP4 recipe's UE4M3 scale (positive inputs)
//
// Build: nvcc -O2 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared -o libcvt_fp4.so cvt_fp4.cu

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__global__ void cvt_e2m1x2_f32_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const float* x = reinterpret_cast<const float*>(in) + 8 * (size_t)t;
    uint32_t d;
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %1, %2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %3, %4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, %5, %6;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, %7, %8;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(d)
        : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
    out[t] = d;
}

__global__ void cvt_f16x2_e2m1x2_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint32_t d;
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "mov.b32 {byte0, byte1, byte2, byte3}, %1;\n"
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
        "}\n"
        : "=r"(d)
        : "r"(in[t] & 0xFFu));
    out[t] = d;
}

__global__ void cvt_ue8m0x2_f32_rz_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint16_t d;
    asm volatile("cvt.rz.satfinite.ue8m0x2.f32 %0, %1, %2;\n" : "=h"(d) : "f"(__uint_as_float(in[2 * t])), "f"(__uint_as_float(in[2 * t + 1])));
    out[t] = d;
}

__global__ void cvt_ue8m0x2_f32_rp_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint16_t d;
    asm volatile("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;\n" : "=h"(d) : "f"(__uint_as_float(in[2 * t])), "f"(__uint_as_float(in[2 * t + 1])));
    out[t] = d;
}

__global__ void cvt_bf16x2_ue8m0x2_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint32_t d;
    const uint16_t a = (uint16_t)(in[t] & 0xFFFFu);
    asm volatile("cvt.rn.bf16x2.ue8m0x2 %0, %1;\n" : "=r"(d) : "h"(a));
    out[t] = d;
}

__global__ void cvt_e4m3x2_f32_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint16_t d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(d) : "f"(__uint_as_float(in[2 * t])), "f"(__uint_as_float(in[2 * t + 1])));
    out[t] = d;
}

}  // namespace

extern "C" {

// op: see the header.  in: u32 words (op 0: 8 per output; ops 2, 3, 5: 2 per output; ops 1, 4: 1), out: u32[n].
int cvt_run(int op, const uint32_t* in, uint32_t* out, int n) {
    static const int per_out[6] = {8, 1, 2, 2, 1, 2};
    if (op < 0 || op > 5 || n < 1) return (int)cudaErrorInvalidValue;
    const size_t in_bytes = (size_t)n * per_out[op] * 4, out_bytes = (size_t)n * 4;
    uint32_t *dIn = nullptr, *dOut = nullptr;
    cudaError_t err;
    if ((err = cudaMalloc(&dIn, in_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dOut, out_bytes)) != cudaSuccess) return (int)err;
    cudaMemcpy(dIn, in, in_bytes, cudaMemcpyHostToDevice);
    const int threads = 256, blocks = (n + threads - 1) / threads;
    switch (op) {
        case 0: cvt_e2m1x2_f32_kernel<<<blocks, threads>>>(dIn, dOut, n); break;
        case 1: cvt_f16x2_e2m1x2_kernel<<<blocks, threads>>>(dIn, dOut, n); break;
        case 2: cvt_ue8m0x2_f32_rz_kernel<<<blocks, threads>>>(dIn, dOut, n); break;
        case 3: cvt_ue8m0x2_f32_rp_kernel<<<blocks, threads>>>(dIn, dOut, n); break;
        case 4: cvt_bf16x2_ue8m0x2_kernel<<<blocks, threads>>>(dIn, dOut, n); break;
        case 5: cvt_e4m3x2_f32_kernel<<<blocks, threads>>>(dIn, dOut, n); break;
    }
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) cudaMemcpy(out, dOut, out_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dIn);
    cudaFree(dOut);
    return (int)err;
}

int device_identity(int* major, int* minor, char* name, int name_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    *major = prop.major;
    *minor = prop.minor;
    int i = 0;
    for (; i < name_len - 1 && prop.name[i]; ++i) name[i] = prop.name[i];
    name[i] = 0;
    return 0;
}

int device_uuid(char* out, int out_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    if (out_len < 41) return (int)cudaErrorInvalidValue;
    static const char* hex = "0123456789abcdef";
    const unsigned char* u = (const unsigned char*)prop.uuid.bytes;
    int p = 0;
    out[p++] = 'G'; out[p++] = 'P'; out[p++] = 'U'; out[p++] = '-';
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out[p++] = '-';
        out[p++] = hex[u[i] >> 4];
        out[p++] = hex[u[i] & 15];
    }
    out[p] = 0;
    return 0;
}

int has_mxf4(void) { return 1; }

}  // extern "C"
