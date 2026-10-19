// RT4-SHEET job 3, width probe: what global/shared memory widths sm_120a emits in one warp instruction, for RT4-SHEET-1's
// claim that a 256-bit global store/load halves the memory count of the bulk reduce-XOR fold. Build and read the SASS; no
// device run. Each kernel's body is one timed loop so cuobjdump shows the store/load opcode and its width.
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v width_probe.cu -o /tmp/wp && cuobjdump -sass /tmp/wp
#include <stdint.h>

__device__ __forceinline__ void st_v4(uint32_t *p, const uint32_t *v) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
}

// does PTX accept a 256-bit vector global store (v8)?  if ptxas rejects it the build names the form
__device__ __forceinline__ void st_v8(uint32_t *p, const uint32_t *v) {
#if defined(RT4_V8)
  asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(p), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]),
               "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]) : "memory");
#else
  st_v4(p, v);
  st_v4(p + 4, v + 4);
#endif
}

__device__ __forceinline__ void ld_v4(const uint32_t *p, uint32_t *v) {
  asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(p) : "memory");
}
__device__ __forceinline__ void ld_v8(const uint32_t *p, uint32_t *v) {
#if defined(RT4_V8)
  asm volatile("ld.global.cg.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]),
               "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]) : "l"(p) : "memory");
#else
  ld_v4(p, v);
  ld_v4(p + 4, v + 4);
#endif
}

// one timed loop a kernel: the store/load's SASS opcode and width is what we read
extern "C" __global__ void probe_st_v4(uint32_t *g, const uint32_t *in, int n) {
  uint32_t v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = in[threadIdx.x * 4 + j];
  for (int s = 0; s < n; ++s) { v[0] += s; st_v4(g + blockIdx.x * 128 + threadIdx.x * 4, v); }
}
extern "C" __global__ void probe_st_v8(uint32_t *g, const uint32_t *in, int n) {
  uint32_t v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = in[threadIdx.x * 8 + j];
  for (int s = 0; s < n; ++s) { v[0] += s; st_v8(g + blockIdx.x * 256 + threadIdx.x * 8, v); }
}
extern "C" __global__ void probe_ld_v8(const uint32_t *g, uint32_t *out, int n) {
  uint32_t v[8] = {};
  for (int s = 0; s < n; ++s) { ld_v8(g + blockIdx.x * 256 + threadIdx.x * 8, v); out[threadIdx.x] ^= v[0] + s; }
}
