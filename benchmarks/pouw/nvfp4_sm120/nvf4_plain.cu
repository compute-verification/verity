// The plain NVFP4 GEMM on mainloop_nvf4.cuh, as entries of the PoUW harness's baseline registry (cutlass_api.h's phc_entry,
// kind 5): D (m x n, row-major BF16) = A B^T, A (m x k) and B (n x k) K-major E2M1 codes with UE4M3 scales per 16 in the
// 128 x 4 layout CUTLASS's and cuBLASLt's NVFP4 GEMMs read; each element BLACKWELL_SM120_NVF4's FP32 chain rounded once to
// BF16 (cvt.rn), as theirs.  To build it into libpouw_harness.so: compile it beside cutlass_registry.cu with
// -I<this directory> -I<cutlass_api.h's>, and add phc_table_nvf4 (and phc_table_fp8) to the registry's tables.
//
// 256 x 128 tiles, persistent (one CTA an SM, tiles in raster groups of 8 tile rows); m a multiple of 256, n of 128, k of 128.
// The workspace holds the two CUtensorMaps (256 bytes, 64-byte aligned), written by prepare.
//
// phc_table_fp8: the plain FP8 E4M3 GEMM on the same mainloop (its FP8 mode), kind 1 (per-tensor scales, all 1, which it
// doesn't read, as CUTLASS's kind-1 kernels): one code a byte, each element BLACKWELL_SM120_E4M3_M16N8K32's FP32 chain
// rounded once to BF16; k a multiple of 64.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>

#include "cutlass_api.h"
#include "mainloop_nvf4.cuh"

namespace {

constexpr int TM = 256, STAGES = 3, RP = 40, GROUP = 8;

template <int BO, int LATE, int WIDE, int FP8>
__global__ void __launch_bounds__(nvf4::Mainloop<TM, STAGES, RP, BO, LATE, WIDE, FP8>::NT, 1)
    nvf4_plain(const nvf4::Maps* maps, const uint8_t* sfa, const uint8_t* sfb, uint16_t* D, int M, int N, int K) {
  using L = nvf4::Mainloop<TM, STAGES, RP, BO, LATE, WIDE, FP8>;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem);
  ml.roles(maps, sfa, sfb, M, N, K, GROUP);
  const int tiles_m = M / TM, tiles_n = N / L::BN, tiles = tiles_m * tiles_n;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    int tm, tn;
    nvf4::raster(t, tiles_m, tiles_n, GROUP, tm, tn);
    float acc[4][4][2][4];
    ml.tile(K, acc, t + (int)gridDim.x < tiles);
    ml.store_bf16(D, N, tm, tn, acc);
  }
}

using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                            const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                            CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

Encode encoder() {
  static Encode fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    return cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess &&
                   q == cudaDriverEntryPointSuccess
               ? reinterpret_cast<Encode>(p)
               : nullptr;
  }();
  return fn;
}

// rows x bytes, row-major: box_rows x 64-byte boxes, SWIZZLE_64B
bool tmap(CUtensorMap* m, const void* ptr, int bytes, int rows, int box_rows) {
  const cuuint64_t dims[2] = {(cuuint64_t)bytes, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)bytes};
  const cuuint32_t box[2] = {64, (cuuint32_t)box_rows}, estr[2] = {1, 1};
  Encode fn = encoder();
  return fn && fn(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int BO, int LATE, int WIDE, int FP8>
struct Prepared : phc_prepared {
  using L = nvf4::Mainloop<TM, STAGES, RP, BO, LATE, WIDE, FP8>;
  const nvf4::Maps* maps;
  phc_args a;
  int grid;
  int launch(void* stream) override {
    nvf4_plain<BO, LATE, WIDE, FP8><<<grid, L::NT, L::SMEM, static_cast<cudaStream_t>(stream)>>>(
        maps, static_cast<const uint8_t*>(a.sfa), static_cast<const uint8_t*>(a.sfb), static_cast<uint16_t*>(a.D), a.m, a.n, a.k);
    return 0;
  }
};

int query(int m, int n, int k, size_t* ws, size_t* sfa_bytes, size_t* sfb_bytes) {
  if (m <= 0 || n <= 0 || k <= 0 || m % TM || n % 128 || k % 128) return 1;
  *ws = sizeof(nvf4::Maps);
  *sfa_bytes = (size_t)m * k / 16;
  *sfb_bytes = (size_t)n * k / 16;
  return 0;
}
int query_fp8(int m, int n, int k, size_t* ws, size_t* sfa_bytes, size_t* sfb_bytes) {
  if (m <= 0 || n <= 0 || k <= 0 || m % TM || n % 128 || k % 64) return 1;
  *ws = sizeof(nvf4::Maps);
  *sfa_bytes = *sfb_bytes = 0;
  return 0;
}

template <int BO, int LATE, int WIDE, int FP8>
int prepare(const phc_args* a, void* ws, size_t ws_bytes, void* stream, phc_prepared** out) {
  size_t need, sa, sb;
  if ((FP8 ? query_fp8 : query)(a->m, a->n, a->k, &need, &sa, &sb) || ws_bytes < need || (reinterpret_cast<uintptr_t>(ws) & 63) ||
      (reinterpret_cast<uintptr_t>(a->A) & 15) || (reinterpret_cast<uintptr_t>(a->B) & 15) ||
      (!FP8 && ((reinterpret_cast<uintptr_t>(a->sfa) & 15) || (reinterpret_cast<uintptr_t>(a->sfb) & 15))))
    return 2;
  alignas(64) CUtensorMap host[2];
  const int bytes = FP8 ? a->k : a->k / 2;
  if (!tmap(&host[0], a->A, bytes, a->m, TM) || !tmap(&host[1], a->B, bytes, a->n, 128)) return 3;
  static_assert(sizeof(host) == sizeof(nvf4::Maps), "two 128-byte CUtensorMaps");
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (cudaMemcpyAsync(ws, host, sizeof(host), cudaMemcpyHostToDevice, s) != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess)
    return 4;
  using L = nvf4::Mainloop<TM, STAGES, RP, BO, LATE, WIDE, FP8>;
  if (cudaFuncSetAttribute(nvf4_plain<BO, LATE, WIDE, FP8>, cudaFuncAttributeMaxDynamicSharedMemorySize, L::SMEM) != cudaSuccess)
    return 5;
  int dev = 0, sms = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
    return 6;
  auto* p = new Prepared<BO, LATE, WIDE, FP8>;
  p->maps = static_cast<const nvf4::Maps*>(ws);
  p->a = *a;
  const int tiles = (a->m / TM) * (a->n / 128);
  p->grid = tiles < sms ? tiles : sms;
  *out = p;
  return 0;
}

// _o: B-major MMA order inside a 64-deep step (CUTLASS's); the other A-major.  _e: LATE (a warp that finds the next stage
// not yet full issues its step's other MMAs before it waits); _ew: LATE and WIDE (a warp's 64 columns contiguous, its BF16
// stores 128-byte rows).  The harness times them all and keeps the fastest.
const phc_entry TABLE[] = {
    {"verity_nvf4_256x128", 5, 0, query, prepare<0, 0, 0, 0>},
    {"verity_nvf4_256x128_o", 5, 0, query, prepare<1, 0, 0, 0>},
    {"verity_nvf4_256x128_e", 5, 0, query, prepare<0, 1, 0, 0>},
    {"verity_nvf4_256x128_o_e", 5, 0, query, prepare<1, 1, 0, 0>},
    {"verity_nvf4_256x128_ew", 5, 0, query, prepare<0, 1, 1, 0>},
    {"verity_nvf4_256x128_o_ew", 5, 0, query, prepare<1, 1, 1, 0>},
};
// the FP8 entries, named as nvf4_bench.cu's kernels
const phc_entry TABLE_FP8[] = {
    {"verity_fp8_256x128", 1, 0, query_fp8, prepare<0, 0, 0, 1>},
    {"verity_fp8_256x128_o_e", 1, 0, query_fp8, prepare<1, 1, 0, 1>},
    {"verity_fp8_256x128_ew", 1, 0, query_fp8, prepare<0, 1, 1, 1>},
    {"verity_fp8_256x128_o_ew", 1, 0, query_fp8, prepare<1, 1, 1, 1>},
};
constexpr int N_NVF4 = int(sizeof(TABLE) / sizeof(TABLE[0])), N_FP8 = int(sizeof(TABLE_FP8) / sizeof(TABLE_FP8[0]));

}  // namespace

const phc_entry* phc_table_nvf4(int* count) {
  *count = N_NVF4;
  return TABLE;
}
const phc_entry* phc_table_fp8(int* count) {
  *count = N_FP8;
  return TABLE_FP8;
}

#ifdef NVF4_PLAIN_ABI
// Standalone, cutlass_registry.cu's C ABI over these tables alone (NVFP4's entries, then FP8's), so nvf4_bench.py and
// fp8_bench.py can gate and time the entries as the harness calls them (PLAIN_LIB):
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared -DNVF4_PLAIN_ABI \
//        -I<cutlass_api.h's directory> nvf4_plain.cu -o libnvf4_plain.so
extern "C" {
static const phc_entry& entry(int i) { return i < N_NVF4 ? TABLE[i] : TABLE_FP8[i - N_NVF4]; }
const char* ph_last_error() { return ""; }
int ph_cutlass_count() { return N_NVF4 + N_FP8; }
int ph_cutlass_describe(int i, const char** name, int* kind, int* swap) {
  if (i < 0 || i >= ph_cutlass_count()) return -1;
  *name = entry(i).name;
  *kind = entry(i).kind;
  *swap = entry(i).swap;
  return 0;
}
int ph_cutlass_query(int i, int m, int n, int k, size_t* ws, size_t* sfa_bytes, size_t* sfb_bytes) {
  if (i < 0 || i >= ph_cutlass_count()) return -1;
  return entry(i).query(m, n, k, ws, sfa_bytes, sfb_bytes);
}
int ph_cutlass_prepare(int i, int m, int n, int k, const void* A, const void* B, const void* sfa, const void* sfb, void* D,
                       void* ws, size_t ws_bytes, void* stream, void** out) {
  if (i < 0 || i >= ph_cutlass_count()) return -1;
  phc_args a{m, n, k, A, B, sfa, sfb, D};
  phc_prepared* p = nullptr;
  const int r = entry(i).prepare(&a, ws, ws_bytes, stream, &p);
  *out = p;
  return r;
}
int ph_cutlass_launch(void* h, void* stream) {
  const int r = static_cast<phc_prepared*>(h)->launch(stream);
  if (r) return r;
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 1000 + int(e);
}
int ph_cutlass_release(void* h) {
  delete static_cast<phc_prepared*>(h);
  return 0;
}
}
#endif
