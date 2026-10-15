// Texture-unit census (round 3): filtered fetch throughput of tex2D / tex3D on a cudaTextureObject over a cudaArray,
// for 8-bit unorm, FP16 and FP32 texels, 1 and 4 channels, bilinear (2D, 4 texels a channel) and trilinear (3D, 8),
// with point sampling as the unfiltered reference; and the filter's weight resolution.
//
// Throughput: 4 blocks of 256 threads a SM, each thread fetching U texels an iteration at coordinates that move every
// iteration (normalized, wrap) and sit between texels, so every fetch filters; the texture (64 x 64, or 32^3) stays in
// the SM's texture cache. The fetched channels are all summed, so none is dead. Each block records its SM cycles and
// %globaltimer span; a fetch's rate is fetches over (SMs x span x the SM clock the blocks' cycles over span give).
//
// Weight resolution: a 2 x 1 FP32 texture (0, 1) read bilinearly at 4,096 points between the texel centres; the
// distinct outputs and their spacing give the weight's fractional bits. The same for 8-bit (0, 255) and FP16 (0, 1).
//
// Prints one JSON line a case.
#include "common.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <cstring>
#include <set>
#include <string>

__device__ __forceinline__ float sum4(float v) { return v; }
__device__ __forceinline__ float sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }

constexpr int U = 8;

template <class T, int DIM>
__global__ void __launch_bounds__(256) k_tex(cudaTextureObject_t t, int iters, float step, float* sink, BlockClk* rec) {
  const int tid = threadIdx.x;
  float x = ((tid & 15) + 0.37f) / 64.f, y = (((tid >> 4) & 15) + 0.61f) / 64.f,
        z = ((blockIdx.x & 7) * 4 + 0.29f) / 32.f;
  float acc = 0.f;
  CLK_BEGIN
#pragma unroll 1
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int k = 0; k < U; k++) {
      if (DIM == 2)
        acc += sum4(tex2D<T>(t, x + k * (1.f / 64.f), y));
      else
        acc += sum4(tex3D<T>(t, x + k * (1.f / 32.f), y, z));
    }
    x += step;
    y += 0.5f * step;
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + tid] = acc;
}

__global__ void k_sweep(cudaTextureObject_t t, int n, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2D<float>(t, 0.5f + (float)i / n, 0.5f);  // unnormalized: texel 0's centre to texel 1's
}

static int g_sms;
static BlockClk* g_rec;
static float* g_sink;

struct Fmt {
  const char* name;
  cudaChannelFormatDesc desc;
  int bytes;  // a texel
  int ch;
  bool normalized_read;  // 8-bit unorm: cudaReadModeNormalizedFloat
};

static void fill(std::vector<uint8_t>& h, const Fmt& f, size_t texels) {
  h.resize(texels * f.bytes);
  uint64_t s = 0x7e57ull;
  for (size_t i = 0; i < texels; i++)
    for (int c = 0; c < f.ch; c++) {
      uint32_t r = (uint32_t)splitmix64(s);
      uint8_t* p = &h[i * f.bytes + c * (f.bytes / f.ch)];
      if (f.bytes / f.ch == 1) *p = (uint8_t)r;
      else if (f.bytes / f.ch == 2) { __half v = __float2half((r & 0xffff) / 65536.f); memcpy(p, &v, 2); }
      else { float v = (r & 0xffffff) / 16777216.f; memcpy(p, &v, 4); }
    }
}

static cudaTextureObject_t make_tex(const Fmt& f, int dim, bool linear, cudaArray_t* arr_out) {
  std::vector<uint8_t> h;
  cudaArray_t arr;
  if (dim == 2) {
    fill(h, f, 64 * 64);
    CK(cudaMallocArray(&arr, &f.desc, 64, 64));
    CK(cudaMemcpy2DToArray(arr, 0, 0, h.data(), 64 * f.bytes, 64 * f.bytes, 64, cudaMemcpyHostToDevice));
  } else {
    fill(h, f, 32 * 32 * 32);
    CK(cudaMalloc3DArray(&arr, &f.desc, make_cudaExtent(32, 32, 32)));
    cudaMemcpy3DParms p = {};
    p.srcPtr = make_cudaPitchedPtr(h.data(), 32 * f.bytes, 32, 32);
    p.dstArray = arr;
    p.extent = make_cudaExtent(32, 32, 32);
    p.kind = cudaMemcpyHostToDevice;
    CK(cudaMemcpy3D(&p));
  }
  cudaResourceDesc rd = {};
  rd.resType = cudaResourceTypeArray;
  rd.res.array.array = arr;
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeWrap;
  td.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
  td.readMode = f.normalized_read ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
  td.normalizedCoords = 1;
  cudaTextureObject_t t;
  CK(cudaCreateTextureObject(&t, &rd, &td, nullptr));
  *arr_out = arr;
  return t;
}

template <class T, int DIM>
static void run_case(const Fmt& f, bool linear, int iters) {
  cudaArray_t arr;
  cudaTextureObject_t t = make_tex(f, DIM, linear, &arr);
  const int blocks = g_sms * 4, threads = 256;
  auto launch = [&] { k_tex<T, DIM><<<blocks, threads>>>(t, iters, 0.0137f, g_sink, g_rec); };
  launch();
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  double best = 0, best_mhz = 0, best_span = 0;
  std::vector<BlockClk> h(blocks);
  for (int r = 0; r < 3; r++) {
    launch();
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), g_rec, blocks * sizeof(BlockClk), cudaMemcpyDeviceToHost));
    long long g0 = h[0].t0_ns, g1 = h[0].t1_ns;
    double mhz_sum = 0;
    for (auto& b : h) {
      g0 = std::min(g0, b.t0_ns), g1 = std::max(g1, b.t1_ns);
      mhz_sum += (double)b.cyc / (double)(b.t1_ns - b.t0_ns) * 1e3;
    }
    double span = (g1 - g0) * 1e-9, mhz = mhz_sum / blocks;
    double per_s = (double)blocks * threads * iters * U / span;
    if (per_s > best) best = per_s, best_mhz = mhz, best_span = span;
  }
  const int taps = linear ? (DIM == 2 ? 4 : 8) : 1;
  const double per_sm_clk = best / g_sms / (best_mhz * 1e6);
  printf("{\"probe\":\"tex\",\"case\":\"%s.%s.%dd\",\"format\":\"%s\",\"channels\":%d,\"dim\":%d,\"filter\":\"%s\","
         "\"taps_a_channel\":%d,\"fetches_per_s\":%.5g,\"sm_mhz\":%.1f,\"fetches_per_sm_per_clk\":%.4f,"
         "\"mac_equiv_per_fetch\":%d,\"mac_equiv_per_s\":%.5g,\"mac_equiv_per_sm_per_clk\":%.3f,\"span_s\":%.4f,"
         "\"sms\":%d,\"blocks\":%d,\"threads\":%d,\"fetches_a_thread\":%d}\n",
         f.name, linear ? (DIM == 2 ? "bilinear" : "trilinear") : "point", DIM, f.name, f.ch, DIM,
         linear ? "linear" : "point", taps, best, best_mhz, per_sm_clk, linear ? taps * f.ch : 0,
         linear ? best * taps * f.ch : 0.0, linear ? per_sm_clk * taps * f.ch : 0.0, best_span, g_sms, blocks, threads,
         iters * U);
  fflush(stdout);
  CK(cudaDestroyTextureObject(t));
  CK(cudaFreeArray(arr));
}

static void sweep(const Fmt& f) {
  const int n = 4096;
  std::vector<uint8_t> h(2 * f.bytes, 0);
  if (f.bytes == 1) h[1] = 255;
  else if (f.bytes == 2) { __half v = __float2half(1.f); memcpy(&h[2], &v, 2); }
  else { float v = 1.f; memcpy(&h[4], &v, 4); }
  cudaArray_t arr;
  CK(cudaMallocArray(&arr, &f.desc, 2, 1));
  CK(cudaMemcpy2DToArray(arr, 0, 0, h.data(), 2 * f.bytes, 2 * f.bytes, 1, cudaMemcpyHostToDevice));
  cudaResourceDesc rd = {};
  rd.resType = cudaResourceTypeArray;
  rd.res.array.array = arr;
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
  td.filterMode = cudaFilterModeLinear;
  td.readMode = f.normalized_read ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
  td.normalizedCoords = 0;
  cudaTextureObject_t t;
  CK(cudaCreateTextureObject(&t, &rd, &td, nullptr));
  float* d;
  CK(cudaMalloc(&d, n * sizeof(float)));
  k_sweep<<<(n + 255) / 256, 256>>>(t, n, d);
  CK(cudaGetLastError());
  std::vector<float> o(n);
  CK(cudaMemcpy(o.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  std::set<float> distinct(o.begin(), o.end());
  std::vector<float> v(distinct.begin(), distinct.end());
  double min_step = 1, max_step = 0, max_off_grid = 0;
  for (size_t i = 1; i < v.size(); i++) min_step = std::min(min_step, (double)v[i] - v[i - 1]),
                                         max_step = std::max(max_step, (double)v[i] - v[i - 1]);
  for (float x : v) max_off_grid = std::max(max_off_grid, std::fabs(x * 256.0 - std::round(x * 256.0)));
  printf("{\"probe\":\"tex_weights\",\"format\":\"%s\",\"points\":%d,\"distinct_outputs\":%zu,\"min_step\":%.6g,"
         "\"max_step\":%.6g,\"fraction_bits\":%.2f,\"max_off_1_256_grid\":%.3g,\"out_at_quarter\":%.8g,"
         "\"out_first\":%.8g,\"out_last\":%.8g}\n",
         f.name, n, distinct.size(), min_step, max_step, std::log2(1.0 / max_step), max_off_grid, o[n / 4], o[0], o[n - 1]);
  fflush(stdout);
  CK(cudaDestroyTextureObject(t));
  CK(cudaFreeArray(arr));
  CK(cudaFree(d));
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 2000;
  g_sms = num_sms();
  CK(cudaMalloc(&g_rec, g_sms * 4 * sizeof(BlockClk)));
  CK(cudaMalloc(&g_sink, g_sms * 4 * 256 * sizeof(float)));
  const Fmt u8x1{"u8x1", cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned), 1, 1, true};
  const Fmt u8x4{"u8x4", cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned), 4, 4, true};
  const Fmt f16x1{"f16x1", cudaCreateChannelDescHalf(), 2, 1, false};
  const Fmt f16x4{"f16x4", cudaCreateChannelDescHalf4(), 8, 4, false};
  const Fmt f32x1{"f32x1", cudaCreateChannelDesc<float>(), 4, 1, false};
  const Fmt f32x4{"f32x4", cudaCreateChannelDesc<float4>(), 16, 4, false};
  run_case<float, 2>(u8x1, true, iters);
  run_case<float4, 2>(u8x4, true, iters);
  run_case<float, 2>(f16x1, true, iters);
  run_case<float4, 2>(f16x4, true, iters);
  run_case<float, 2>(f32x1, true, iters);
  run_case<float4, 2>(f32x4, true, iters);
  run_case<float4, 2>(u8x4, false, iters);
  run_case<float4, 2>(f16x4, false, iters);
  run_case<float4, 3>(u8x4, true, iters);
  run_case<float4, 3>(f16x4, true, iters);
  run_case<float, 3>(u8x1, true, iters);
  sweep(u8x1);
  sweep(f16x1);
  sweep(f32x1);
  return 0;
}
