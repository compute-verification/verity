// The PoUW honest GEMMs on the RTX 4090 (sm_89), as a torch extension (benchmarks/pouw/kernels/__init__.py builds it).
//
//   ncp_gemm(A s8 [M,K], B s8 [N,K], mode, cfg) -> C s32 [M,N] (+ checkpoints)   C = A·Bᵀ with an m16n8k16 s8 mma chain
//     mode 0: bare; 1: every accumulator register read once per 16-deep step and folded (run = run·odd + acc: one IMAD
//     per word, the free hash feed's stand-in); 2: every checkpoint written out (int32 [M, N, K/16], for the bit-identity
//     gate at small shapes).
//   ncp_route_u(Xu u8 [M,K], Y s8 [N,K], cfg) -> C s32 [M,N], checkpoints s32 [M,N,K/16]
//     c0[j] = -128*sum_l Y[j,l]; each m16n8k16 u8xs8 step is dumped.  The Python wrapper hashes 16x16 tiles
//     step-major with ncp-v1's SHAKE256 framing.
//   fp8_gemm(A e4m3 [M,K], B e4m3 [N,K], cfg) -> C f32 [M,N]: m16n8k32 e4m3 mma with FP32 accumulate, a fresh +0
//     accumulator every 128 products promoted to the total by one FP32 add (the `ada` device of verity.core.protocols.pouw's
//     pearl-fp8-v4, which the gate checks bit for bit).
//
// The tiling is the PoUW research store's code/pouw-gpu-constants/ncp_gemm.cu (cp.async multistage pipeline, ldmatrix.x4
// from a swizzled tile): M, N and K must be multiples of the config's tile and of 64.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#define BK 64
__device__ __forceinline__ void cp_async16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ uint32_t swz(int r, int ch) { return r * 64 + ((ch ^ ((r >> 1) & 3)) << 4); }

__device__ __forceinline__ void mma16(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma16_u8s8(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#define NRUN 8

// MODE: 0 bare, 1 fold, 2 dump (signed NCP); 3 fp8 windowed (Pearl ada); 4 native route-U dump.
template <int MT, int NT, int WM, int WN, int STAGES, int MODE>
__global__ void __launch_bounds__(WM * WN * 32)
    k_gemm(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, void* __restrict__ Cout, uint32_t* __restrict__ runout,
           int32_t* __restrict__ ckpt, const int32_t* __restrict__ c0, int M, int N, int K) {
  constexpr int NTHR = WM * WN * 32, BM = WM * MT * 16, BN = WN * NT * 8;
  constexpr bool FP8 = MODE == 3, ROUTE_U = MODE == 4;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sA = (uint32_t)__cvta_generic_to_shared(smem), sB = sA + STAGES * BM * BK;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
  const int nbn = N / BN, bm = blockIdx.x / nbn, bn = blockIdx.x % nbn;
  const uint8_t* gA = A + (size_t)bm * BM * K;
  const uint8_t* gB = B + (size_t)bn * BN * K;
  const int KT = K / BK, T = K / 16;
  auto load = [&](int stage, int kt) {
#pragma unroll
    for (int i = 0; i < (BM * 4 + NTHR - 1) / NTHR; i++) {
      int idx = tid + i * NTHR, r = idx >> 2, ch = idx & 3;
      if (idx < BM * 4) cp_async16(sA + stage * BM * BK + swz(r, ch), gA + (size_t)r * K + kt * BK + ch * 16);
    }
#pragma unroll
    for (int i = 0; i < (BN * 4 + NTHR - 1) / NTHR; i++) {
      int idx = tid + i * NTHR, r = idx >> 2, ch = idx & 3;
      if (idx < BN * 4) cp_async16(sB + stage * BN * BK + swz(r, ch), gB + (size_t)r * K + kt * BK + ch * 16);
    }
  };
  const int g = lane >> 2, tg = lane & 3;
  uint32_t c[MT][NT][4], run[NRUN];
  float fc[FP8 ? MT : 1][FP8 ? NT : 1][4], ft[FP8 ? MT : 1][FP8 ? NT : 1][4];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if constexpr (ROUTE_U) {
          int col = bn * BN + wn * NT * 8 + nt * 8 + tg * 2;
          c[mt][nt][i] = c0[col + (i & 1)];
        } else {
          c[mt][nt][i] = 0;
        }
        if (FP8) { fc[FP8 ? mt : 0][FP8 ? nt : 0][i] = 0.f; ft[FP8 ? mt : 0][FP8 ? nt : 0][i] = 0.f; }
      }
#pragma unroll
  for (int j = 0; j < NRUN; j++) run[j] = j;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { if (s < KT) load(s, s); cp_commit(); }
  const int a_row = wm * MT * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), a_ch = lane >> 4;
  const int b_row = wn * NT * 8 + (lane >> 4) * 8 + (lane & 7), b_ch = (lane >> 3) & 1;
#pragma unroll 1
  for (int kt = 0; kt < KT; kt++) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    { int nk = kt + STAGES - 1; if (nk < KT) load(nk % STAGES, nk); cp_commit(); }
    const int st = kt % STAGES;
    const uint32_t as = sA + st * BM * BK, bs = sB + st * BN * BK;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      uint32_t af[MT][4], bf[NT][2];
#pragma unroll
      for (int mt = 0; mt < MT; mt++) ldsm4(af[mt], as + swz(a_row + mt * 16, kk * 2 + a_ch));
#pragma unroll
      for (int nt = 0; nt < NT; nt += 2) ldsm4(&bf[nt][0], bs + swz(b_row + nt * 8, kk * 2 + b_ch));
      if constexpr (FP8) {
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
          for (int nt = 0; nt < NT; nt++) mma_e4m3(fc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
      } else {
#pragma unroll
        for (int h = 0; h < 2; h++) {
#pragma unroll
          for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
              if constexpr (ROUTE_U) mma16_u8s8(c[mt][nt], af[mt][2 * h], af[mt][2 * h + 1], bf[nt][h]);
              else mma16(c[mt][nt], af[mt][2 * h], af[mt][2 * h + 1], bf[nt][h]);
            }
          if constexpr (MODE == 1) {
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
              for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) { uint32_t& r = run[(nt * 4 + i) % NRUN]; r = r * 0x9E3779B1u + c[mt][nt][i]; }
          }
          if constexpr (MODE == 2 || MODE == 4) {
            const int t = kt * 4 + kk * 2 + h;
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
              for (int nt = 0; nt < NT; nt++) {
                size_t row = (size_t)bm * BM + wm * MT * 16 + mt * 16 + g, col = (size_t)bn * BN + wn * NT * 8 + nt * 8 + tg * 2;
                ckpt[((row * N + col) * T) + t] = c[mt][nt][0];
                ckpt[((row * N + col + 1) * T) + t] = c[mt][nt][1];
                ckpt[(((row + 8) * N + col) * T) + t] = c[mt][nt][2];
                ckpt[(((row + 8) * N + col + 1) * T) + t] = c[mt][nt][3];
              }
          }
        }
      }
    }
    if constexpr (FP8) {
      if ((kt & 1) || kt == KT - 1) {        // every 128 products (two 64-deep stages), and the last partial window
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
          for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
              float t = __fadd_rn(ft[mt][nt][i], fc[mt][nt][i]);
              ft[mt][nt][i] = t == 0.f ? 0.f : t;
              fc[mt][nt][i] = 0.f;
            }
      }
    }
  }
  cp_wait<0>();
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      size_t row = (size_t)bm * BM + wm * MT * 16 + mt * 16 + g, col = (size_t)bn * BN + wn * NT * 8 + nt * 8 + tg * 2;
      if constexpr (FP8) {
        float* C = reinterpret_cast<float*>(Cout);
        *reinterpret_cast<float2*>(C + row * N + col) = make_float2(ft[mt][nt][0], ft[mt][nt][1]);
        *reinterpret_cast<float2*>(C + (row + 8) * N + col) = make_float2(ft[mt][nt][2], ft[mt][nt][3]);
      } else {
        int32_t* C = reinterpret_cast<int32_t*>(Cout);
        *reinterpret_cast<uint2*>(C + row * N + col) = make_uint2(c[mt][nt][0], c[mt][nt][1]);
        *reinterpret_cast<uint2*>(C + (row + 8) * N + col) = make_uint2(c[mt][nt][2], c[mt][nt][3]);
      }
    }
  if constexpr (MODE == 1) {
    uint32_t s = 0;
#pragma unroll
    for (int j = 0; j < NRUN; j++) s ^= run[j];
    runout[(size_t)blockIdx.x * NTHR + tid] = s;
  }
}

__global__ void route_u_c0(const uint8_t* __restrict__ B, int32_t* __restrict__ c0, int N, int K) {
  int j = blockIdx.x;
  int sum = 0;
  for (int l = threadIdx.x; l < K; l += blockDim.x) sum += (int)(int8_t)B[(size_t)j * K + l];
  __shared__ int sums[256];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) sums[threadIdx.x] += sums[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0) c0[j] = (int32_t)(0u - 128u * (uint32_t)sums[0]);
}

// Configs: (MT, NT, WM, WN, STAGES).  0: 128 x 256; 1: 16 x 256; 2: 16 x 16 route-U tile;
// FP8 uses 0 -> 128 x 128 and 1 -> 16 x 128 below.
template <int MT, int NT, int WM, int WN, int STAGES, int MODE>
static void launch(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, torch::Tensor& run,
                   torch::Tensor& ck, const torch::Tensor& c0) {
  constexpr int NTHR = WM * WN * 32, BM = WM * MT * 16, BN = WN * NT * 8;
  const int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0, "shape (", M, ", ", K, ", ", N, ") is not a multiple of the tile ", BM, " x ", BN, " x ", BK);
  auto kern = k_gemm<MT, NT, WM, WN, STAGES, MODE>;
  size_t smem = (size_t)STAGES * (BM + BN) * BK;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int blocks = (M / BM) * (N / BN);
  if (MODE == 1 && run.numel() < (int64_t)blocks * NTHR) run.resize_({(int64_t)blocks * NTHR});
  kern<<<blocks, NTHR, smem, c10::cuda::getCurrentCUDAStream()>>>(A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), C.data_ptr(),
      (uint32_t*)run.data_ptr<int32_t>(), ck.numel() ? ck.data_ptr<int32_t>() : nullptr,
      c0.numel() ? c0.data_ptr<int32_t>() : nullptr, M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> ncp_gemm(torch::Tensor A, torch::Tensor B, int64_t mode, int64_t cfg) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kUInt8 && B.dtype() == torch::kUInt8 && A.is_contiguous() && B.is_contiguous());
  const int64_t M = A.size(0), K = A.size(1), N = B.size(0);
  auto C = torch::empty({M, N}, A.options().dtype(torch::kInt32));
  auto run = torch::empty({1}, A.options().dtype(torch::kInt32));
  auto ck = mode == 2 ? torch::empty({M, N, K / 16}, A.options().dtype(torch::kInt32)) : torch::empty({0}, A.options().dtype(torch::kInt32));
  auto c0 = torch::empty({0}, A.options().dtype(torch::kInt32));
#define DISPATCH(MD)                                                                  \
  if (cfg == 0) launch<4, 8, 2, 4, 4, MD>(A, B, C, run, ck, c0);                       \
  else if (cfg == 1) launch<1, 8, 1, 4, 4, MD>(A, B, C, run, ck, c0);                  \
  else launch<1, 2, 1, 1, 3, MD>(A, B, C, run, ck, c0);
  if (mode == 0) { DISPATCH(0) } else if (mode == 1) { DISPATCH(1) } else { DISPATCH(2) }
#undef DISPATCH
  return {C, run, ck};
}

std::vector<torch::Tensor> ncp_route_u(torch::Tensor A, torch::Tensor B, int64_t cfg) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kUInt8 && B.dtype() == torch::kUInt8
              && A.is_contiguous() && B.is_contiguous());
  TORCH_CHECK(A.size(1) == B.size(1), "route-U operands have the same stacked depth");
  const int64_t M = A.size(0), K = A.size(1), N = B.size(0);
  auto C = torch::empty({M, N}, A.options().dtype(torch::kInt32));
  auto ck = torch::empty({M, N, K / 16}, A.options().dtype(torch::kInt32));
  auto run = torch::empty({1}, A.options().dtype(torch::kInt32));
  auto c0 = torch::empty({N}, A.options().dtype(torch::kInt32));
  route_u_c0<<<N, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      B.data_ptr<uint8_t>(), c0.data_ptr<int32_t>(), N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (cfg == 0) launch<4, 8, 2, 4, 4, 4>(A, B, C, run, ck, c0);
  else if (cfg == 1) launch<1, 8, 1, 4, 4, 4>(A, B, C, run, ck, c0);
  else launch<1, 2, 1, 1, 3, 4>(A, B, C, run, ck, c0);
  return {C, ck};
}

torch::Tensor fp8_gemm(torch::Tensor A, torch::Tensor B, int64_t cfg) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kUInt8 && B.dtype() == torch::kUInt8 && A.is_contiguous() && B.is_contiguous());
  auto C = torch::empty({A.size(0), B.size(0)}, A.options().dtype(torch::kFloat32));
  auto run = torch::empty({1}, A.options().dtype(torch::kInt32));
  auto ck = torch::empty({0}, A.options().dtype(torch::kInt32));
  auto c0 = torch::empty({0}, A.options().dtype(torch::kInt32));
  if (cfg == 0) launch<4, 4, 2, 4, 3, 3>(A, B, C, run, ck, c0);
  else launch<1, 4, 1, 4, 4, 3>(A, B, C, run, ck, c0);
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ncp_gemm", &ncp_gemm, "NCP's s8 GEMM with 16-deep checkpoint reads (mode 0 bare, 1 fold, 2 dump)");
  m.def("ncp_route_u", &ncp_route_u, "NCP route U: u8 x s8, c0 = -128*colsum(Y), every d=16 checkpoint dumped");
  m.def("fp8_gemm", &fp8_gemm, "Pearl's e4m3 GEMM on the ada device (promote every 128 products)");
}
