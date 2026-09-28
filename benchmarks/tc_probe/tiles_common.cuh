// Host side shared by the tile-probe translation units (mma_tiles.cu, wgmma_tiles.cu): the copy-in / launch / copy-out of one
// batch of tiles and the device identity.  A tile is A[M][K] and Bt[8][K] operand words (raw bits, k contiguous) with C[M][8]
// and D[M][8] FP32 words; every output word is the hardware's own function of (C[i][j], A[i][:], Bt[j][:]).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace tc_probe {

typedef void (*launch_fn)(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles);

// Copies ntiles tiles to the device, runs `launch` once over all of them and copies D back.  Returns the cudaError_t as an int.
inline int run_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles, size_t a_bytes_per_tile,
                     size_t b_bytes_per_tile, size_t c_words_per_tile, launch_fn launch) {
    const size_t a_bytes = (size_t)ntiles * a_bytes_per_tile;
    const size_t b_bytes = (size_t)ntiles * b_bytes_per_tile;
    const size_t c_bytes = (size_t)ntiles * c_words_per_tile * 4;
    uint8_t *dA = nullptr, *dB = nullptr;
    uint32_t *dC = nullptr, *dD = nullptr;
    cudaError_t err = cudaSuccess;
    if ((err = cudaMalloc(&dA, a_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dB, b_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dC, c_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dD, c_bytes)) != cudaSuccess) return (int)err;
    cudaMemcpy(dA, A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bt, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, c_bytes, cudaMemcpyHostToDevice);
    cudaMemset(dD, 0, c_bytes);
    launch(dA, dB, dC, dD, ntiles);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) cudaMemcpy(D, dD, c_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);
    return (int)err;
}

}  // namespace tc_probe

extern "C" {

// The device's compute capability (major, minor) and its name; for the capture's provenance.
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

}  // extern "C"
