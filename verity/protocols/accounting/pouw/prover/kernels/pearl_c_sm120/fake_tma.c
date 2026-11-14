// The CPU stand-in for libcuda.so.1 (../pearl_c/fake_cuda.c, with the opt-in limit and the static shared memory that
// run.py's pd_setup reads) with the entry point the mainloop's host side adds,
// cuTensorMapEncodeTiled.  It refuses what mainloop_sm120.cuh doesn't take (not UINT8, rank 2, SWIZZLE_128B, L2 promotion
// 128B, no interleave, no fill; a box other than 128 bytes by 1..256 rows; element strides other than 1) and what the driver
// refuses (a map not 64-byte aligned, an address not 16-byte aligned, a row stride that isn't a multiple of 16 or isn't the
// depth, a matrix outside one allocation), logs {"op":"tmap","ptr","depth","rows","box"} to $FAKE_LOG, and writes the
// address, depth and rows into the map's first 24 bytes.
#include "../pearl_c/fake_cuda.c"

typedef unsigned long long cuuint64_t;
typedef unsigned int cuuint32_t;

CUresult CUDAAPI cuTensorMapEncodeTiled(void* map, int dtype, cuuint32_t rank, void* addr, const cuuint64_t* dims,
                                        const cuuint64_t* strides, const cuuint32_t* box, const cuuint32_t* estr, int interleave,
                                        int swizzle, int l2, int oob) {
  if (!map || (uintptr_t)map % 64 || (uintptr_t)addr % 16 || dtype || rank != 2 || interleave || swizzle != 3 || l2 != 2 || oob)
    return CUDA_ERROR_INVALID_VALUE;
  if (strides[0] != dims[0] || strides[0] % 16 || box[0] != 128 || !box[1] || box[1] > 256 || estr[0] != 1 || estr[1] != 1)
    return CUDA_ERROR_INVALID_VALUE;
  if (!find((uintptr_t)addr, (size_t)(dims[0] * dims[1]))) return CUDA_ERROR_INVALID_VALUE;
  FILE* L = log_file();
  fprintf(L, "{\"op\":\"tmap\",\"ptr\":%llu,\"depth\":%llu,\"rows\":%llu,\"box\":[%u,%u]}\n", (unsigned long long)(uintptr_t)addr,
          dims[0], dims[1], box[0], box[1]);
  fflush(L);
  const uint64_t head[3] = {(uint64_t)(uintptr_t)addr, dims[0], dims[1]};
  memset(map, 0, 128);
  memcpy(map, head, sizeof head);
  return CUDA_SUCCESS;
}
