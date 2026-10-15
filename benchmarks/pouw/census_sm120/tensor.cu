// Tensor-pipe census (item 1): for every mma.sync kind sm_120a accepts, the sustained rate (MACs a SM a clock), the
// issue cost (sub-partition cycles a warp instruction occupies the tensor pipe at saturation) and the dependent latency.
//
// Throughput: one block a SM (60 KB of dynamic shared memory keeps a second one off), W warps, each with NACC
// independent accumulators over register-resident operands (two operand sets per lane, valid codes, no NaN), the loop
// rolled so its SASS body is exactly NACC MMAs. Every block times its own loop with clock64 after a barrier, so the rate
// is per SM per SM clock whatever clock the card runs at. Latency: one warp, one accumulator chain.
// `census.py` reads the SASS of each kernel's loop to name the instruction ptxas emitted (QMMA, OMMA, HMMA, IMMA, DMMA,
// or an emulation).
#include "common.cuh"

#include <cstring>
#include <string>

struct F32x4 { float v[4]; };
struct S32x4 { int v[4]; };
struct U32x2 { uint32_t v[2]; };
struct F64x2 { double v[2]; };

__device__ __forceinline__ void zero(F32x4& d) { d.v[0] = d.v[1] = d.v[2] = d.v[3] = 0.f; }
__device__ __forceinline__ void zero(S32x4& d) { d.v[0] = d.v[1] = d.v[2] = d.v[3] = 0; }
__device__ __forceinline__ void zero(U32x2& d) { d.v[0] = d.v[1] = 0; }
__device__ __forceinline__ void zero(F64x2& d) { d.v[0] = d.v[1] = 0.0; }
__device__ __forceinline__ uint32_t fold(const F32x4& d) {
  return __float_as_uint(d.v[0]) ^ __float_as_uint(d.v[1]) ^ __float_as_uint(d.v[2]) ^ __float_as_uint(d.v[3]);
}
__device__ __forceinline__ uint32_t fold(const S32x4& d) { return d.v[0] ^ d.v[1] ^ d.v[2] ^ d.v[3]; }
__device__ __forceinline__ uint32_t fold(const U32x2& d) { return d.v[0] ^ d.v[1]; }
__device__ __forceinline__ uint32_t fold(const F64x2& d) {
  return (uint32_t)__double_as_longlong(d.v[0]) ^ (uint32_t)__double_as_longlong(d.v[1]);
}

// Operand word formats, one letter a word (host fill): 8 = E4M3 bytes, 5 = E5M2 bytes, 4 = E2M1 nibbles, 6 = FP6 in
// the low 6 bits of each byte, i = int bytes, h = FP16x2, b = BF16x2, t = TF32, d/D = an FP64's low/high word,
// s = UE4M3 scales (1.0), S = UE8M0 scales (1.0), m = 2:4 metadata (pairs {0,1}), r = raw.
#define D4 "{%0,%1,%2,%3}"
#define F4OUT "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])
#define S4OUT "+r"(d.v[0]), "+r"(d.v[1]), "+r"(d.v[2]), "+r"(d.v[3])
#define A4 "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
#define B2 "r"(b[0]), "r"(b[1])
#define B4 "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3])

#define KIND(NAME, ACC, NA_, NB_, MACS_, STORED_, FA, FB, PTX, BODY) \
  struct NAME {                                                   \
    using Acc = ACC;                                              \
    static constexpr int NA = NA_, NB = NB_;                      \
    static constexpr double MACS = MACS_, STORED = STORED_;       \
    static constexpr const char* name = #NAME;                    \
    static constexpr const char* ptx = PTX;                       \
    static constexpr const char* fa = FA;                         \
    static constexpr const char* fb = FB;                         \
    __device__ __forceinline__ static void mma(Acc& d, const uint32_t* a, const uint32_t* b) { BODY; } \
  };

// Dense A4/B2 with an FP32 accumulator.
#define BODY_F32_A4B2(P) asm volatile(P " " D4 ", {%4,%5,%6,%7}, {%8,%9}, " D4 ";" : F4OUT : A4, B2)
#define BODY_S32_A4B2(P) asm volatile(P " " D4 ", {%4,%5,%6,%7}, {%8,%9}, " D4 ";" : S4OUT : A4, B2)
#define BODY_F16_A4B2(P) \
  asm volatile(P " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" : "+r"(d.v[0]), "+r"(d.v[1]) : A4, B2)
// Block-scaled: a[4] and b[2] are the scale words.
#define BODY_F32_A4B2_SC(P) \
  asm volatile(P " " D4 ", {%4,%5,%6,%7}, {%8,%9}, " D4 ", {%10}, {0, 0}, {%11}, {0, 0};" : F4OUT : A4, B2, "r"(a[4]), "r"(b[2]))
// 2:4 sparse: A4 compressed, B4, a[4] the metadata, selector 0.
#define BODY_F32_SP(P) \
  asm volatile(P " " D4 ", {%4,%5,%6,%7}, {%8,%9,%10,%11}, " D4 ", %12, 0x0;" : F4OUT : A4, B4, "r"(a[4]))
#define BODY_S32_SP(P) \
  asm volatile(P " " D4 ", {%4,%5,%6,%7}, {%8,%9,%10,%11}, " D4 ", %12, 0x0;" : S4OUT : A4, B4, "r"(a[4]))
#define BODY_F32_SP_SC(P)                                                                               \
  asm volatile(P " " D4 ", {%4,%5,%6,%7}, {%8,%9,%10,%11}, " D4 ", %12, 0x0, {%13}, {0, 0}, {%14}, {0, 0};" \
               : F4OUT : A4, B4, "r"(a[4]), "r"(a[5]), "r"(b[4]))

#define P_E4M3_F32 "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
#define P_E4M3_F16 "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"
#define P_E5M2_F32 "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"
#define P_E4M3E5M2 "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32"
#define P_F8F6F4_E2M1 "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32"
#define P_F8F6F4_E3M2 "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32"
#define P_MXF8_E4M3 "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0"
#define P_NVFP4 "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"
#define P_MXFP4 "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"
#define P_MXF4 "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"
#define P_SP_E4M3 "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32"
#define P_SP_NVFP4 \
  "mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3"
#define P_SP_MXF8 \
  "mma.sp::ordered_metadata.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k64.row.col.f32.e4m3.e4m3.f32.ue8m0"
#define P_S8 "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
#define P_SP_S8 "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32"
#define P_S4 "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32"
#define P_B1 "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc"
#define P_BF16 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
#define P_F16_F32 "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
#define P_F16_F16 "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
#define P_SP_BF16 "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"
#define P_TF32 "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
#define P_SP_TF32 "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32"
#define P_F16_K8 "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
#define P_F64 "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"

KIND(e4m3_f32, F32x4, 4, 2, 4096, 4096, "8888", "88", P_E4M3_F32, BODY_F32_A4B2(P_E4M3_F32))
KIND(e4m3_f16acc, U32x2, 4, 2, 4096, 4096, "8888", "88", P_E4M3_F16, BODY_F16_A4B2(P_E4M3_F16))
KIND(e5m2_f32, F32x4, 4, 2, 4096, 4096, "5555", "55", P_E5M2_F32, BODY_F32_A4B2(P_E5M2_F32))
KIND(e4m3xe5m2_f32, F32x4, 4, 2, 4096, 4096, "8888", "55", P_E4M3E5M2, BODY_F32_A4B2(P_E4M3E5M2))
KIND(f8f6f4_e2m1, F32x4, 4, 2, 4096, 4096, "4444", "44", P_F8F6F4_E2M1, BODY_F32_A4B2(P_F8F6F4_E2M1))
KIND(f8f6f4_e3m2, F32x4, 4, 2, 4096, 4096, "6666", "66", P_F8F6F4_E3M2, BODY_F32_A4B2(P_F8F6F4_E3M2))
KIND(mxf8_e4m3, F32x4, 5, 3, 4096, 4096, "8888S", "88S", P_MXF8_E4M3, BODY_F32_A4B2_SC(P_MXF8_E4M3))
KIND(nvfp4, F32x4, 5, 3, 8192, 8192, "4444s", "44s", P_NVFP4, BODY_F32_A4B2_SC(P_NVFP4))
KIND(mxfp4, F32x4, 5, 3, 8192, 8192, "4444S", "44S", P_MXFP4, BODY_F32_A4B2_SC(P_MXFP4))
KIND(mxf4, F32x4, 5, 3, 8192, 8192, "4444S", "44S", P_MXF4, BODY_F32_A4B2_SC(P_MXF4))
KIND(sp_e4m3_f32, F32x4, 5, 4, 8192, 4096, "8888m", "8888", P_SP_E4M3, BODY_F32_SP(P_SP_E4M3))
KIND(sp_nvfp4, F32x4, 6, 5, 16384, 8192, "4444ms", "4444s", P_SP_NVFP4, BODY_F32_SP_SC(P_SP_NVFP4))
KIND(sp_mxf8_e4m3, F32x4, 6, 5, 8192, 4096, "8888mS", "8888S", P_SP_MXF8, BODY_F32_SP_SC(P_SP_MXF8))
KIND(s8_s32, S32x4, 4, 2, 4096, 4096, "iiii", "ii", P_S8, BODY_S32_A4B2(P_S8))
KIND(sp_s8_s32, S32x4, 5, 4, 8192, 4096, "iiiim", "iiii", P_SP_S8, BODY_S32_SP(P_SP_S8))
KIND(s4_s32, S32x4, 4, 2, 8192, 8192, "rrrr", "rr", P_S4, BODY_S32_A4B2(P_S4))
KIND(b1_and_popc, S32x4, 4, 2, 32768, 32768, "rrrr", "rr", P_B1, BODY_S32_A4B2(P_B1))
KIND(bf16_f32, F32x4, 4, 2, 2048, 2048, "bbbb", "bb", P_BF16, BODY_F32_A4B2(P_BF16))
KIND(f16_f32, F32x4, 4, 2, 2048, 2048, "hhhh", "hh", P_F16_F32, BODY_F32_A4B2(P_F16_F32))
KIND(f16_f16acc, U32x2, 4, 2, 2048, 2048, "hhhh", "hh", P_F16_F16, BODY_F16_A4B2(P_F16_F16))
KIND(sp_bf16_f32, F32x4, 5, 4, 4096, 2048, "bbbbm", "bbbb", P_SP_BF16, BODY_F32_SP(P_SP_BF16))
KIND(tf32_f32, F32x4, 4, 2, 1024, 1024, "tttt", "tt", P_TF32, BODY_F32_A4B2(P_TF32))
KIND(sp_tf32_f32, F32x4, 5, 4, 2048, 1024, "ttttm", "tttt", P_SP_TF32, BODY_F32_SP(P_SP_TF32))
KIND(f16_k8_f32, F32x4, 2, 1, 1024, 1024, "hh", "h", P_F16_K8,
     asm volatile(P_F16_K8 " " D4 ", {%4,%5}, {%6}, " D4 ";" : F4OUT : "r"(a[0]), "r"(a[1]), "r"(b[0])))
KIND(f64_m8n8k4, F64x2, 2, 2, 256, 256, "dD", "dD", P_F64,
     double av = __hiloint2double(a[1], a[0]); double bv = __hiloint2double(b[1], b[0]);
     asm volatile(P_F64 " {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d.v[0]), "+d"(d.v[1]) : "d"(av), "d"(bv)))

constexpr int FRAG_WORDS = 32;  // per lane: A set 0 at 0, A set 1 at 8, B set 0 at 16, B set 1 at 24

template <class K, int NACC, int MAXT>
__global__ void __launch_bounds__(MAXT) k_mma(const uint32_t* __restrict__ fr, long long iters, uint32_t* __restrict__ sink,
                                               BlockClk* rec) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint32_t* f = fr + ((warp & 7) * 32 + lane) * FRAG_WORDS;
  uint32_t a0[K::NA], a1[K::NA], b0[K::NB], b1[K::NB];
#pragma unroll
  for (int i = 0; i < K::NA; i++) { a0[i] = f[i]; a1[i] = f[8 + i]; }
#pragma unroll
  for (int i = 0; i < K::NB; i++) { b0[i] = f[16 + i]; b1[i] = f[24 + i]; }
  typename K::Acc d[NACC];
#pragma unroll
  for (int c = 0; c < NACC; c++) zero(d[c]);
  CLK_BEGIN
#pragma unroll 1
  for (long long i = 0; i < iters; i++) {
#pragma unroll
    for (int c = 0; c < NACC; c++) K::mma(d[c], (c & 1) ? a1 : a0, (c & 2) ? b1 : b0);
  }
  CLK_END(rec)
  uint32_t x = 0;
#pragma unroll
  for (int c = 0; c < NACC; c++) x ^= fold(d[c]);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

static uint32_t fill_word(char fmt, uint64_t& s) {
  uint32_t w = (uint32_t)splitmix64(s);
  switch (fmt) {
    case '8': return w & 0xBFBFBFBFu;               // E4M3: exponent <= 7, no NaN
    case '5': return w & 0xBFBFBFBFu;               // E5M2: exponent <= 15, no Inf/NaN
    case '4': return w;                             // E2M1: every code finite
    case '6': return w & 0x3F3F3F3Fu;               // FP6 in the low 6 bits of each byte
    case 'i': return w;
    case 'h': return (w & 0x83FF83FFu) | 0x3C003C00u;  // FP16 in [1, 2) with random sign
    case 'b': return (w & 0x807F807Fu) | 0x3F803F80u;  // BF16 in [1, 2)
    case 't': return (w & 0x807FE000u) | 0x3F800000u;  // TF32 in [1, 2)
    case 'd': return w;
    case 'D': return (w & 0x800FFFFFu) | 0x3FF00000u;  // FP64 in [1, 2)
    case 's': return 0x38383838u;                   // UE4M3 1.0
    case 'S': return 0x7F7F7F7Fu;                   // UE8M0 1.0
    case 'm': return 0x44444444u;                   // 2:4 metadata, elements {0, 1} of every group
    default: return w;
  }
}

static uint32_t* g_frags = nullptr;
static uint32_t* g_sink = nullptr;
static BlockClk* g_rec = nullptr;
static int g_sms = 0;

template <class K>
static void fill_frags() {
  std::vector<uint32_t> h(8 * 32 * FRAG_WORDS, 0);
  uint64_t s = 0x5eed0001ull;
  for (int w = 0; w < 8; w++)
    for (int l = 0; l < 32; l++) {
      uint32_t* f = &h[(w * 32 + l) * FRAG_WORDS];
      for (int i = 0; i < K::NA; i++) { f[i] = fill_word(K::fa[i], s); f[8 + i] = fill_word(K::fa[i], s); }
      for (int i = 0; i < K::NB; i++) { f[16 + i] = fill_word(K::fb[i], s); f[24 + i] = fill_word(K::fb[i], s); }
    }
  CK(cudaMemcpy(g_frags, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
}

template <class K, int NACC, int MAXT>
static Timing launch_mma(int blocks, int warps, long long iters) {
  auto kern = k_mma<K, NACC, MAXT>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  return run_timed([&] { kern<<<blocks, warps * 32, ONE_BLOCK_SMEM>>>(g_frags, iters, g_sink, g_rec); }, blocks, g_rec);
}

template <class K, int NACC, int MAXT>
static void throughput(int warps, double target_cycles) {
  Timing t0 = launch_mma<K, NACC, MAXT>(g_sms, warps, 64);
  double per_iter = t0.cyc_med / 64.0;
  long long iters = std::max(64LL, (long long)(target_cycles / per_iter));
  Timing t = launch_mma<K, NACC, MAXT>(g_sms, warps, iters);
  double instr_per_sm = (double)warps * NACC * iters;
  double macs_clk = instr_per_sm * K::MACS / t.cyc_med;
  printf("{\"probe\":\"tensor\",\"mode\":\"throughput\",\"kind\":\"%s\",\"ptx\":\"%s\",\"macs_per_instr\":%.0f,"
         "\"stored_macs_per_instr\":%.0f,\"warps\":%d,\"nacc\":%d,\"iters\":%lld,",
         K::name, K::ptx, K::MACS, K::STORED, warps, NACC, iters);
  print_timing(t);
  printf(",\"macs_per_sm_per_clk\":%.3f,\"stored_macs_per_sm_per_clk\":%.3f,\"warp_instr_per_sm_per_clk\":%.5f,"
         "\"smsp_cycles_per_instr\":%.4f,\"device_tmacs_per_s\":%.2f}\n",
         macs_clk, instr_per_sm * K::STORED / t.cyc_med, instr_per_sm / t.cyc_med, 4.0 * t.cyc_med / instr_per_sm,
         instr_per_sm * g_sms * K::MACS / (t.ms * 1e-3) / 1e12);
  fflush(stdout);
}

template <class K>
static void latency() {
  const long long iters = 4096;
  Timing t = launch_mma<K, 1, 1024>(1, 1, iters);
  printf("{\"probe\":\"tensor\",\"mode\":\"latency\",\"kind\":\"%s\",\"iters\":%lld,", K::name, iters);
  print_timing(t);
  printf(",\"cycles_per_dependent_instr\":%.3f}\n", t.cyc_med / iters);
  fflush(stdout);
}

template <class K>
static void census(double target_cycles, const char* only) {
  if (only && !strstr(only, K::name)) return;
  fill_frags<K>();
  latency<K>();
  throughput<K, 8, 512>(4, target_cycles);
  throughput<K, 8, 512>(8, target_cycles);
  throughput<K, 8, 512>(16, target_cycles);
  throughput<K, 4, 1024>(32, target_cycles);
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  double target = argc > 2 ? atof(argv[2]) : 3e7;  // SM cycles a timed launch (~15 ms at 2.1 GHz)
  g_sms = num_sms();
  CK(cudaMalloc(&g_frags, 8 * 32 * FRAG_WORDS * 4));
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  census<e4m3_f32>(target, only);
  census<e4m3_f16acc>(target, only);
  census<e5m2_f32>(target, only);
  census<e4m3xe5m2_f32>(target, only);
  census<f8f6f4_e2m1>(target, only);
  census<f8f6f4_e3m2>(target, only);
  census<mxf8_e4m3>(target, only);
  census<nvfp4>(target, only);
  census<mxfp4>(target, only);
  census<mxf4>(target, only);
  census<sp_e4m3_f32>(target, only);
  census<sp_nvfp4>(target, only);
  census<sp_mxf8_e4m3>(target, only);
  census<s8_s32>(target, only);
  census<sp_s8_s32>(target, only);
  census<s4_s32>(target / 8, only);
  census<b1_and_popc>(target / 8, only);
  census<bf16_f32>(target, only);
  census<f16_f32>(target, only);
  census<f16_f16acc>(target, only);
  census<sp_bf16_f32>(target, only);
  census<tf32_f32>(target, only);
  census<sp_tf32_f32>(target, only);
  census<f16_k8_f32>(target, only);
  census<f64_m8n8k4>(target / 4, only);
  printf("{\"probe\":\"tensor\",\"done\":true}\n");
  return 0;
}
