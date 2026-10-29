// FOLD-DYN's honest steps on sm_90a: one kernel a family, each folding 8 independent states over 64 unrolled steps,
// so `fold_dyn_sass.sh` reads a step's lane instructions off the SASS. A step's word is w_t = w_{t-1} + d[t] (the
// zero-data chain, one IADD3); a step's coins (rotation r_t, multiplier k_t, keys x_t, y_t) are read from constant
// memory, one set a step, as a call's live coins would be. The `imm` kernels take r_t as a template immediate, as a
// kernel specialized to a call's draw would.
#include <cstdint>

#define STEPS 64
#define LANES 8

__constant__ uint32_t K[STEPS];
__constant__ uint32_t X[STEPS];
__constant__ uint32_t Y[STEPS];
__constant__ uint32_t R[STEPS];
__constant__ uint32_t D[STEPS];

__device__ __forceinline__ uint32_t rotl(uint32_t x, uint32_t r) { return __funnelshift_l(x, x, r); }

template <int F>
__device__ __forceinline__ uint32_t step(uint32_t s, uint32_t w, int t) {
    if (F == 0) return rotl(s, R[t]) + w;                    // rotadd, register rotation
    if (F == 1) return rotl(s ^ X[t], R[t]) + w;             // rotxoradd
    if (F == 2) return rotl(s, R[t]) * K[t] + w;             // rotmuladd
    if (F == 3) return rotl(s * K[t], R[t]) + w;             // mulrotadd
    if (F == 4) return (s ^ w) * K[t];                       // xormul, SPARSE's (c4)
    if (F == 5) return rotl(s ^ w, R[t]);                    // xorrot
    if (F == 6) return (s ^ w ^ X[t]) * K[t] + Y[t];         // lik, the live-keyed li half step
    if (F == 7) return (rotl(s ^ X[t], R[t]) + w) * K[t] + Y[t];  // arxm
    return 0;
}

// the rotation an immediate, fixed at compile time: (7t + 3) mod 31 + 1; G 0 rotadd, 1 mulrotadd, 2 arxm
template <int G, int T>
__device__ __forceinline__ void imm_steps(uint32_t (&s)[LANES], uint32_t (&w)[LANES]) {
    if constexpr (T < STEPS) {
        constexpr int r = (7 * T + 3) % 31 + 1;
#pragma unroll
        for (int l = 0; l < LANES; l++) {
            w[l] += D[T];
            if constexpr (G == 0) s[l] = __funnelshift_l(s[l], s[l], r) + w[l];
            if constexpr (G == 1) {
                uint32_t m = s[l] * K[T];
                s[l] = __funnelshift_l(m, m, r) + w[l];
            }
            if constexpr (G == 2) {
                uint32_t u = s[l] ^ X[T];
                s[l] = (__funnelshift_l(u, u, r) + w[l]) * K[T] + Y[T];
            }
        }
        imm_steps<G, T + 1>(s, w);
    }
}

template <int F>
__global__ void fold(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    uint32_t s[LANES], w[LANES];
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * LANES;
#pragma unroll
    for (int l = 0; l < LANES; l++) { s[l] = in[i + l]; w[l] = in[i + l] * 0x9E3779B9u; }
#pragma unroll
    for (int t = 0; t < STEPS; t++)
#pragma unroll
        for (int l = 0; l < LANES; l++) {
            w[l] += D[t];
            s[l] = step<F>(s[l], w[l], t);
        }
#pragma unroll
    for (int l = 0; l < LANES; l++) out[i + l] = s[l] ^ w[l];
}

template <int G>
__global__ void fold_imm(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    uint32_t s[LANES], w[LANES];
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * LANES;
#pragma unroll
    for (int l = 0; l < LANES; l++) { s[l] = in[i + l]; w[l] = in[i + l] * 0x9E3779B9u; }
    imm_steps<G, 0>(s, w);
#pragma unroll
    for (int l = 0; l < LANES; l++) out[i + l] = s[l] ^ w[l];
}

template __global__ void fold<0>(const uint32_t*, uint32_t*);
template __global__ void fold<1>(const uint32_t*, uint32_t*);
template __global__ void fold<2>(const uint32_t*, uint32_t*);
template __global__ void fold<3>(const uint32_t*, uint32_t*);
template __global__ void fold<4>(const uint32_t*, uint32_t*);
template __global__ void fold<5>(const uint32_t*, uint32_t*);
template __global__ void fold<6>(const uint32_t*, uint32_t*);
template __global__ void fold<7>(const uint32_t*, uint32_t*);
template __global__ void fold_imm<0>(const uint32_t*, uint32_t*);
template __global__ void fold_imm<1>(const uint32_t*, uint32_t*);
template __global__ void fold_imm<2>(const uint32_t*, uint32_t*);
