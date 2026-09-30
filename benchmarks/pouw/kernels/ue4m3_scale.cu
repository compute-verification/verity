// ue4m3_scale.cu: Pearl-C4's block-scale step on sm_120a (`fp4-design.md` step 4), the reference and cheaper exact variants.
//
//   s_b = max(e4m3.rn.sat(mul.rn(amax_b, f32(1/6))), 2^-9)     the byte the MMA reads, four per 32-bit scale register
//   r_b = div.rn(1, s_b)                                         the cast's multiplier, code = e2m1(f32(D·r_b))
//
// Variants of the reciprocal (the encode is F2FP in all but ienc_lut):
//   ref_rn      decode (cvt.rn.f16x2.e4m3x2, then f16 -> f32) and __frcp_rn: the reference
//   ref_approx  decode and rcp.approx.ftz.f32 (GPU 4's priced path; exact only if MUFU happens to be on these 126 inputs)
//   lut         r_b = lut[byte], a 128-entry FP32 table in shared memory (decode and reciprocal in one load)
//   lut_lane    the same table replicated per lane (lut[byte][lane], 16 KB): conflict-free for any bytes
//   lut256      the per-lane table at a 256-byte stride (32 KB), so one PRMT forms the address (byte << 8) | (lane << 2)
//               straight from the MMA's packed scale register: one PRMT and one LDS per scale, conflict-free
//   exp_mant    an 8-entry mantissa table plus the exponent as an integer, 8 more entries for the subnormal bytes
//   newton      FFMA only: decode, quadratic seed on the mantissa, two Newton steps, the exponent as an integer
//   ienc_lut    the byte from integer ops and one FADD instead of F2FP (2^-9 floor as an integer max), then lut
// ue4m3_scale.py checks each on the CPU; ue_gates checks them on the card.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define DEV __device__ __forceinline__

__constant__ uint32_t c_lut[128];      // RN_f32(1/s) per byte; [0] = 2^9, [127] unused
__constant__ uint32_t c_tab[16];       // [0..7] RN_f32(1/(1 + j/8)); [8..15] the subnormal bytes' reciprocals

constexpr float SIXTH = 0.16666667163372039794921875f;   // f32(1/6) = 0x3E2AAAAB
constexpr float FLOOR = 0x1p-9f;
constexpr float C2 = 0.37409148f, C1 = -1.5990455f, C0 = 2.2202663f;

DEV uint32_t enc2(float lo, float hi) {                     // byte0 = e4m3(lo), byte1 = e4m3(hi)
    uint16_t d;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
    return d;
}

DEV uint32_t encode4(const float a[4]) {                    // the MMA's scale register
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = fmaxf(a[i] * SIXTH, FLOOR);
    return enc2(v[0], v[1]) | (enc2(v[2], v[3]) << 16);
}

DEV void decode4(uint32_t p, float s[4]) {
    uint32_t h01, h23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"((uint16_t)(p & 0xffff)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"((uint16_t)(p >> 16)));
    float2 f01 = __half22float2(*reinterpret_cast<__half2*>(&h01));
    float2 f23 = __half22float2(*reinterpret_cast<__half2*>(&h23));
    s[0] = f01.x; s[1] = f01.y; s[2] = f23.x; s[3] = f23.y;
}

DEV uint32_t ienc(float v) {                                 // e4m3.rn.satfinite for finite v >= 0, no F2FP
    uint32_t u = __float_as_uint(v);
    uint32_t n = (u - (120u << 23) + 0x7ffffu + ((u >> 20) & 1u)) >> 20;
    uint32_t s = __float_as_uint(v + 0x1p14f) - 0x46800000u;
    return min(u < 0x3c800000u ? s : n, 0x7eu);
}

enum Variant { REF_RN = 0, REF_APPROX, LUT, LUT_LANE, EXP_MANT, NEWTON, IENC_LUT, LUT256, BASE, NVARIANT };

DEV float lut256(const float* smem, uint32_t p, int k, uint32_t lane4) {     // smem byte address (p.byte_k << 8) | lane4
    uint32_t addr = __byte_perm(p, lane4, 0x5504u | (uint32_t(k) << 4));
    return *reinterpret_cast<const float*>(reinterpret_cast<const char*>(smem) + addr);
}

template <int V>
DEV uint32_t scale4(const float a[4], float r[4], const float* lut, const uint32_t* tab, int lane) {
    if constexpr (V == BASE) {                               // the 1/6 multiply and the floor only: the timing baseline
#pragma unroll
        for (int i = 0; i < 4; ++i) r[i] = fmaxf(a[i] * SIXTH, FLOOR);
        return __float_as_uint(r[0]);
    } else if constexpr (V == IENC_LUT) {
        uint32_t b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            b[i] = max(ienc(a[i] * SIXTH), 1u);
            r[i] = lut[b[i]];
        }
        return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
    } else {
        uint32_t p = encode4(a);
        if constexpr (V == REF_RN || V == REF_APPROX || V == NEWTON) {
            float s[4];
            decode4(p, s);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (V == REF_RN) {
                    r[i] = __frcp_rn(s[i]);
                } else if constexpr (V == REF_APPROX) {
                    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r[i]) : "f"(s[i]));
                } else {
                    uint32_t u = __float_as_uint(s[i]);
                    float m = __uint_as_float((u & 0x007fffffu) | 0x3f800000u);
                    float y = fmaf(fmaf(C2, m, C1), m, C0);
                    y = fmaf(y, fmaf(-m, y, 1.0f), y);
                    y = fmaf(y, fmaf(-m, y, 1.0f), y);
                    r[i] = __uint_as_float(__float_as_uint(y) - ((u & 0x7f800000u) - (127u << 23)));
                }
            }
        } else if constexpr (V == LUT256) {
#pragma unroll
            for (int i = 0; i < 4; ++i) r[i] = lut256(lut, p, i, uint32_t(lane) << 2);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t b = (p >> (8 * i)) & 0xffu;
                if constexpr (V == LUT) {
                    r[i] = lut[b];
                } else if constexpr (V == LUT_LANE) {
                    r[i] = lut[b * 32 + lane];
                } else {
                    uint32_t e = b >> 3;
                    uint32_t t = tab[e ? (b & 7u) : (b | 8u)];
                    r[i] = __uint_as_float(t + (e ? ((7u - e) << 23) : 0u));
                }
            }
        }
        return p;
    }
}

template <int V>
DEV void fill_smem(float* lut, uint32_t* tab) {
    if constexpr (V == LUT || V == IENC_LUT) {
        for (int i = threadIdx.x; i < 128; i += blockDim.x) lut[i] = __uint_as_float(c_lut[i]);
    } else if constexpr (V == LUT_LANE) {
        for (int i = threadIdx.x; i < 128 * 32; i += blockDim.x) lut[i] = __uint_as_float(c_lut[i >> 5]);
    } else if constexpr (V == LUT256) {
        for (int i = threadIdx.x; i < 128 * 64; i += blockDim.x) lut[i] = __uint_as_float(c_lut[i >> 6]);
    } else if constexpr (V == EXP_MANT) {
        for (int i = threadIdx.x; i < 16; i += blockDim.x) tab[i] = c_tab[i];
    }
    __syncthreads();
}

// Timing: every thread carries 4 groups of 4 block amaxes, perturbed each iteration by a loop-carried FADD so nothing
// hoists; the outputs fold into an integer and a float accumulator. Per iteration: 16 scales, 16 FADD (inputs), 16 FADD
// (fold of r), 4 LOP3 (fold of the scale registers), plus the loop control.
template <int V>
DEV void time_body(const float* __restrict__ seed, uint32_t* __restrict__ out, int iters) {
    extern __shared__ float smem[];
    uint32_t* tab = reinterpret_cast<uint32_t*>(smem);
    fill_smem<V>(smem, tab);
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    float a[4][4];
#pragma unroll
    for (int g = 0; g < 4; ++g)
#pragma unroll
        for (int i = 0; i < 4; ++i) a[g][i] = seed[(t * 16 + g * 4 + i) & 1023];
    const float d = seed[1024];
    uint32_t acc = 0;
    float facc = 0.0f;
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            float r[4];
            acc ^= scale4<V>(a[g], r, smem, tab, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                facc += r[i];
                a[g][i] += d;
            }
        }
    }
    out[t] = acc ^ __float_as_uint(facc);
}

#define TIME_KERNEL(name, V) \
    extern "C" __global__ void __launch_bounds__(256) time_##name(const float* seed, uint32_t* out, int iters) { \
        time_body<V>(seed, out, iters); }
TIME_KERNEL(ref_rn, REF_RN)
TIME_KERNEL(ref_approx, REF_APPROX)
TIME_KERNEL(lut, LUT)
TIME_KERNEL(lut_lane, LUT_LANE)
TIME_KERNEL(exp_mant, EXP_MANT)
TIME_KERNEL(newton, NEWTON)
TIME_KERNEL(ienc_lut, IENC_LUT)
TIME_KERNEL(lut256, LUT256)
TIME_KERNEL(base, BASE)

// Gates. gate_recip: every byte 1..126, each variant's r against __frcp_rn(decode(byte)); bad[v] counts mismatches.
template <int V>
DEV float recip_of_byte(uint32_t b, const float* lut, const uint32_t* tab) {
    float r[4];
    if constexpr (V == LUT || V == IENC_LUT) {
        return lut[b];
    } else if constexpr (V == EXP_MANT) {
        uint32_t e = b >> 3;
        return __uint_as_float(tab[e ? (b & 7u) : (b | 8u)] + (e ? ((7u - e) << 23) : 0u));
    } else {
        float s[4];
        decode4(b, s);
        if constexpr (V == REF_APPROX) {
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r[0]) : "f"(s[0]));
            return r[0];
        } else {
            uint32_t u = __float_as_uint(s[0]);
            float m = __uint_as_float((u & 0x007fffffu) | 0x3f800000u);
            float y = fmaf(fmaf(C2, m, C1), m, C0);
            y = fmaf(y, fmaf(-m, y, 1.0f), y);
            y = fmaf(y, fmaf(-m, y, 1.0f), y);
            return __uint_as_float(__float_as_uint(y) - ((u & 0x7f800000u) - (127u << 23)));
        }
    }
}

extern "C" __global__ void gate_recip(unsigned long long* bad) {
    __shared__ float lut[128];
    __shared__ uint32_t tab[16];
    for (int i = threadIdx.x; i < 128; i += blockDim.x) lut[i] = __uint_as_float(c_lut[i]);
    for (int i = threadIdx.x; i < 16; i += blockDim.x) tab[i] = c_tab[i];
    __syncthreads();
    uint32_t b = threadIdx.x;
    if (b < 1 || b > 126) return;
    float s[4];
    decode4(b, s);
    uint32_t ref = __float_as_uint(__frcp_rn(s[0]));
    if (__float_as_uint(recip_of_byte<REF_APPROX>(b, lut, tab)) != ref) atomicAdd(&bad[REF_APPROX], 1ull);
    if (__float_as_uint(recip_of_byte<LUT>(b, lut, tab)) != ref) atomicAdd(&bad[LUT], 1ull);
    if (__float_as_uint(recip_of_byte<EXP_MANT>(b, lut, tab)) != ref) atomicAdd(&bad[EXP_MANT], 1ull);
    if (__float_as_uint(recip_of_byte<NEWTON>(b, lut, tab)) != ref) atomicAdd(&bad[NEWTON], 1ull);
}

// gate_lut256: every byte 1..126 from every lane through lut256's addressing, against __frcp_rn(decode(byte)).
extern "C" __global__ void gate_lut256(unsigned long long* bad) {
    extern __shared__ float gsmem[];
    fill_smem<LUT256>(gsmem, nullptr);
    uint32_t b = blockIdx.x + 1, lane = threadIdx.x & 31;
    if (b > 126) return;
    float s[4];
    decode4(b, s);
    for (int k = 0; k < 4; ++k)
        if (__float_as_uint(lut256(gsmem, b << (8 * k), k, lane << 2)) != __float_as_uint(__frcp_rn(s[0])))
            atomicAdd(&bad[LUT256], 1ull);
}

// gate_pipeline: every finite non-negative FP32 amax (0 .. 0x7F7FFFFF). bad[0]: the LUT path's r differs from the
// reference's; bad[1]: ienc's floored byte differs from F2FP's; bad[2]: ienc_lut's r differs from the reference's.
extern "C" __global__ void gate_pipeline(unsigned long long* bad) {
    __shared__ float lut[128];
    for (int i = threadIdx.x; i < 128; i += blockDim.x) lut[i] = __uint_as_float(c_lut[i]);
    __syncthreads();
    unsigned long long n0 = 0, n1 = 0, n2 = 0;
    for (uint32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < 0x7f800000u; u += gridDim.x * blockDim.x) {
        float v = fmaxf(__uint_as_float(u) * SIXTH, FLOOR);
        uint32_t b = enc2(v, 0.0f) & 0xffu;
        float s[4];
        decode4(b, s);
        uint32_t ref = __float_as_uint(__frcp_rn(s[0]));
        n0 += __float_as_uint(lut[b]) != ref;
        uint32_t bi = max(ienc(__uint_as_float(u) * SIXTH), 1u);
        n1 += bi != b;
        n2 += __float_as_uint(lut[bi]) != ref;
    }
    if (n0) atomicAdd(&bad[0], n0);
    if (n1) atomicAdd(&bad[1], n1);
    if (n2) atomicAdd(&bad[2], n2);
}

static const void* const KERNELS[NVARIANT] = {(const void*)time_ref_rn, (const void*)time_ref_approx, (const void*)time_lut,
                                              (const void*)time_lut_lane, (const void*)time_exp_mant, (const void*)time_newton,
                                              (const void*)time_ienc_lut, (const void*)time_lut256, (const void*)time_base};

extern "C" int ue_setup(const uint32_t* lut, const uint32_t* tab, int* sms, int* major, int* minor, char* uuid16) {
    if (cudaMemcpyToSymbol(c_lut, lut, sizeof(uint32_t) * 128) != cudaSuccess) return 1;
    if (cudaMemcpyToSymbol(c_tab, tab, sizeof(uint32_t) * 16) != cudaSuccess) return 2;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 3;
    *sms = p.multiProcessorCount; *major = p.major; *minor = p.minor;
    for (int i = 0; i < 16; ++i) uuid16[i] = p.uuid.bytes[i];
    return 0;
}

extern "C" int ue_gates(unsigned long long* host_bad /* [NVARIANT + 3] */) {
    unsigned long long* d;
    if (cudaMalloc(&d, sizeof(unsigned long long) * (NVARIANT + 3)) != cudaSuccess) return 1;
    cudaMemset(d, 0, sizeof(unsigned long long) * (NVARIANT + 3));
    gate_recip<<<1, 128>>>(d);
    gate_lut256<<<126, 32, 128 * 256>>>(d);
    gate_pipeline<<<2048, 256>>>(d + NVARIANT);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(host_bad, d, sizeof(unsigned long long) * (NVARIANT + 3), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return e == cudaSuccess ? 0 : 2;
}

// One launch of variant v: blocks x 256 threads, iters iterations of 16 scales per thread. Returns ms and SM clocks.
extern "C" int ue_time(int v, int blocks, int iters, double* ms, double* clk_khz) {
    if (v < 0 || v >= NVARIANT) return 1;
    float h[1025];
    for (int i = 0; i < 1024; ++i) h[i] = 0.001f + 3000.0f * (float)((i * 2654435761u) % 1000003u) / 1000003.0f;
    h[1024] = 1e-7f;
    float* seed; uint32_t* out;
    cudaMalloc(&seed, sizeof(h));
    cudaMalloc(&out, sizeof(uint32_t) * blocks * 256);
    cudaMemcpy(seed, h, sizeof(h), cudaMemcpyHostToDevice);
    size_t smem = v == LUT256 ? 128 * 256 : v == LUT_LANE ? 128 * 32 * 4 : 512;
    cudaFuncSetAttribute(KERNELS[v], cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    void* args[] = {&seed, &out, &iters};
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaLaunchKernel(KERNELS[v], blocks, 256, args, smem, 0);            // warm-up
    cudaEventRecord(e0);
    cudaLaunchKernel(KERNELS[v], blocks, 256, args, smem, 0);
    cudaEventRecord(e1);
    cudaError_t e = cudaEventSynchronize(e1);
    float t = 0.0f;
    cudaEventElapsedTime(&t, e0, e1);
    *ms = t;
    int khz = 0;
    cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
    *clk_khz = khz;
    cudaFree(seed); cudaFree(out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return e == cudaSuccess ? 0 : 2;
}
