// CP-W's checkpoint (`PouwWarpCheck_v2` at fold fx1) on the served sm_120 unit, by the routes a cheater can
// take that put the merge on the memory system or the uniform datapath, for what each costs in SASS and on the device. A
// unit is one IMMA.16832 (128 words a warp, 4 a lane) and fx1's absorb of its words (`segment._fx1_absorb`: d = p
// fx1_key(q), u = y ^ d ^ KK[t], y = u KA[t] + z, z = rotl(u, ROT[t]) + y, chain 4 (t % 2) + e); a set is 8 units, and
// after every set the warp's checkpoint is two words, the XOR over its 32 lanes and 8 chains of y and of z
// (`segment.warp_checkpoint`). Keys are compile-time constants, as served. A and B are all-ones s8 fragments and D
// accumulates, so word q of set s is W[q] + 32 (s + 1) whatever the fragment layout, and the host recomputes every lane.
// Modes, the checkpoint's route:
//   0 none (the fold alone);
//   1 st128: four 128-bit st.global.v4.b32 of the lane's 16 state words (egress only: no reduction, not a checkpoint);
//   2 redux16: REDUX.XOR of each of the 16 state words, the 8 y and 8 z results XORed where ptxas keeps them (the
//     uniform datapath), lane 0 stores both;
//   3 redg32: the lane's 16 words XORed to two (LOP3), two 32-bit red.global.xor into the zeroed receipt words;
//   4 atoms32: the same into shared memory (red.shared.xor.b32), flushed after the last set;
//   5 redg16: no lane merge, each of the 16 words its own 32-bit red.global.xor into its receipt word;
//   6 atoms16: the same into shared memory, flushed after the last set;
//   7 redux2: the lane's 16 words XORed to two (LOP3), then one REDUX.XOR of each, lane 0 stores both (LI-CP-W's
//     honest route in dc/probes, at fx1's two words);
//   8 foldmem (`cpwm`): no checkpoint, the fold itself with its XOR on the memory system, for FOLD-OPT at π*. A round
//     trip is t = 2h and 2h + 1, one step of each of the 8 chains: every lane stores its 8 y to the warp's global slot
//     and its 8 p fk to the warp's shared slot, lane 0 bulk-reduces (cp.reduce.async.bulk .xor, elementwise) that
//     shared slot and then the static KK slot into the global one, and every lane reads u = y ^ p fk ^ KK back past
//     L1. The lane keeps the two IMADs and the rotate-add;
//   9 foldmem at a block (`cpwb`): mode 8 with the block's slot (uniform addresses), thread 0 issuing both bulk XORs
//     for every warp of the block and the block's barrier in place of the warp's, so the bulk ops' uniform glue is
//     paid once a block;
//   10 foldmem at a 16-warp block (`cpwx`): mode 9 at 4 NT threads, its slots in 80 KB of dynamic shared memory, so
//     the glue is paid once for 16 warps.
// `main` runs on the device it finds: bitexact (one warp, 64 sets, every final state and every checkpoint against the
// host; mode 10 one whole block, every thread's final state), then tp at 1, 2, 4 warps a SMSP and at full occupancy:
// SMSP-cycles a warp-set (clock64 a block, max over blocks), 8 units a warp-set.
#include <stdint.h>
#include <stdio.h>

#define NCH 8
#define NT 128
#define CPS 64

__host__ __device__ constexpr uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
__host__ __device__ constexpr uint32_t KA(int t) { return mix32(0xA0000000u + 2u * t) | 1u; }
__host__ __device__ constexpr uint32_t KK(int t) { return mix32(0xB0000000u + 2u * t + 1u); }
__host__ __device__ constexpr int ROT(int t) { return ((t * 7 + 5) % 31) + 1; }
__host__ __device__ constexpr uint32_t FK(int q) { return mix32(0xA0000000u + 2u * (16u + 32u * q)) | 1u; }

__device__ __forceinline__ void imma(uint32_t (&d)[4], uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a), "r"(b));
}

template <int T, int E>
__device__ __forceinline__ void step(uint32_t (&y)[NCH], uint32_t (&z)[NCH], uint32_t p) {
  constexpr int c = 4 * (T % 2) + E;
  constexpr uint32_t fk = FK(4 * T + E), kk = KK(T), ka = KA(T);
  constexpr int r = ROT(T);
  uint32_t u = y[c] ^ (p * fk) ^ kk;
  y[c] = u * ka + z[c];
  z[c] = __funnelshift_l(u, u, r) + y[c];
}

template <int T>
__device__ __forceinline__ void absorb(uint32_t (&y)[NCH], uint32_t (&z)[NCH], const uint32_t (&d)[4]) {
  step<T, 0>(y, z, d[0]);
  step<T, 1>(y, z, d[1]);
  step<T, 2>(y, z, d[2]);
  step<T, 3>(y, z, d[3]);
}

__device__ __forceinline__ void st128(uint32_t *p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void redg_xor(uint32_t *p, uint32_t v) {
  asm volatile("red.relaxed.gpu.global.xor.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void reds_xor(uint32_t addr, uint32_t v) {
  asm volatile("red.relaxed.cta.shared.xor.b32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}

template <int MODE>
__global__ void __launch_bounds__(NT) cpw(const uint32_t *st, const uint32_t *W, uint32_t *out, uint32_t *cp,
                                          long long *cyc, int sets, uint32_t one) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x, ln = idx & 31, warp = idx >> 5, wb = threadIdx.x >> 5;
  const int src = idx & 1023;
  __shared__ uint32_t scp[NT / 32][2 * CPS];
  uint32_t y[NCH], z[NCH], d[8][4];
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    y[j] = st[src * 16 + j];
    z[j] = st[src * 16 + NCH + j];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[i][e] = W[src * 32 + 4 * i + e];
  if (MODE == 4 || MODE == 6)
    for (int k = ln; k < 2 * CPS; k += 32) scp[wb][k] = 0;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(&scp[wb][0]);
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int s = 0; s < sets; ++s) {
#pragma unroll
    for (int i = 0; i < 8; ++i) imma(d[i], one, one);
    absorb<0>(y, z, d[0]);
    absorb<1>(y, z, d[1]);
    absorb<2>(y, z, d[2]);
    absorb<3>(y, z, d[3]);
    absorb<4>(y, z, d[4]);
    absorb<5>(y, z, d[5]);
    absorb<6>(y, z, d[6]);
    absorb<7>(y, z, d[7]);
    const int c = s & (CPS - 1), k = warp * CPS + c;
    if (MODE == 1) {
      st128(&out[idx * 16], y[0], y[1], y[2], y[3]);
      st128(&out[idx * 16 + 4], y[4], y[5], y[6], y[7]);
      st128(&out[idx * 16 + 8], z[0], z[1], z[2], z[3]);
      st128(&out[idx * 16 + 12], z[4], z[5], z[6], z[7]);
    } else if (MODE == 2) {
      uint32_t ry = 0, rz = 0;
#pragma unroll
      for (int j = 0; j < NCH; ++j) {
        ry ^= __reduce_xor_sync(0xffffffffu, y[j]);
        rz ^= __reduce_xor_sync(0xffffffffu, z[j]);
      }
      if (ln == 0) *(uint2 *)&cp[2 * k] = make_uint2(ry, rz);
    } else if (MODE == 3 || MODE == 4 || MODE == 7) {
      uint32_t xy = y[0], xz = z[0];
#pragma unroll
      for (int j = 1; j < NCH; ++j) {
        xy ^= y[j];
        xz ^= z[j];
      }
      if (MODE == 3) {
        redg_xor(&cp[2 * k], xy);
        redg_xor(&cp[2 * k + 1], xz);
      } else if (MODE == 4) {
        reds_xor(sb + 8u * c, xy);
        reds_xor(sb + 8u * c + 4u, xz);
      } else {
        const uint32_t ry = __reduce_xor_sync(0xffffffffu, xy), rz = __reduce_xor_sync(0xffffffffu, xz);
        if (ln == 0) *(uint2 *)&cp[2 * k] = make_uint2(ry, rz);
      }
    } else if (MODE == 5) {
#pragma unroll
      for (int j = 0; j < NCH; ++j) {
        redg_xor(&cp[2 * k], y[j]);
        redg_xor(&cp[2 * k + 1], z[j]);
      }
    } else if (MODE == 6) {
#pragma unroll
      for (int j = 0; j < NCH; ++j) {
        reds_xor(sb + 8u * c, y[j]);
        reds_xor(sb + 8u * c + 4u, z[j]);
      }
    }
  }
  __syncthreads();
  long long t1 = clock64();
  if (MODE != 1) {
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      out[idx * 16 + j] = y[j];
      out[idx * 16 + NCH + j] = z[j];
    }
  }
  if (MODE == 4 || MODE == 6)
    for (int k = ln; k < 2 * CPS; k += 32) cp[2 * warp * CPS + k] = scp[wb][k];
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

__device__ __forceinline__ void sts64(uint32_t addr, uint32_t a, uint32_t b) {
  asm volatile("st.shared.v2.b32 [%0], {%1,%2};" ::"r"(addr), "r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ void ldcg128(const uint32_t *p, uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
  asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
}
// one bulk XOR of a warp's 32 NCH words, shared -> global, then its completion
__device__ __forceinline__ void bulk_xor(uint32_t *g, uint32_t s) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%1], %2;\n\t"
               "cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group 0;" ::"l"(g), "r"(s), "n"(32 * NCH * 4)
               : "memory");
}

template <int H>
__device__ __forceinline__ void trip(uint32_t (&y)[NCH], uint32_t (&z)[NCH], const uint32_t (&d)[8][4], uint32_t *gy,
                                     uint32_t sd, uint32_t sk, int ln) {
  uint32_t m[NCH];
#pragma unroll
  for (int j = 0; j < NCH; ++j) m[j] = d[2 * H + j / 4][j % 4] * FK(4 * (2 * H + j / 4) + j % 4);
  uint32_t *g = gy + NCH * ln;
  st128(g, y[0], y[1], y[2], y[3]);
  st128(g + 4, y[4], y[5], y[6], y[7]);
  const uint32_t s = sd + 4u * NCH * ln;
  sts64(s, m[0], m[1]);
  sts64(s + 8, m[2], m[3]);
  sts64(s + 16, m[4], m[5]);
  sts64(s + 24, m[6], m[7]);
  asm volatile("fence.proxy.async;" ::: "memory");
  __syncwarp();
  if (ln == 0) {
    bulk_xor(gy, sd);
    bulk_xor(gy, sk);
    asm volatile("fence.proxy.async;" ::: "memory");
  }
  __syncwarp();
  uint32_t u[NCH];
  ldcg128(g, u[0], u[1], u[2], u[3]);
  ldcg128(g + 4, u[4], u[5], u[6], u[7]);
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    y[j] = u[j] * KA(2 * H + j / 4) + z[j];
    z[j] = __funnelshift_l(u[j], u[j], ROT(2 * H + j / 4)) + y[j];
  }
}

// mode 8; `cp` holds each warp's 32 NCH-word y slot
template <int MODE>
__global__ void __launch_bounds__(NT) cpwm(const uint32_t *st, const uint32_t *W, uint32_t *out, uint32_t *cp,
                                           long long *cyc, int sets, uint32_t one) {
  static_assert(MODE == 8, "cpwm is mode 8");
  const int idx = blockIdx.x * blockDim.x + threadIdx.x, ln = idx & 31, warp = idx >> 5, wb = threadIdx.x >> 5;
  const int src = idx & 1023;
  __shared__ __align__(128) uint32_t sd[NT / 32][32 * NCH];
  __shared__ __align__(128) uint32_t sk[4][32 * NCH];
  uint32_t y[NCH], z[NCH], d[8][4];
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    y[j] = st[src * 16 + j];
    z[j] = st[src * 16 + NCH + j];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[i][e] = W[src * 32 + 4 * i + e];
  for (int i = threadIdx.x; i < 4 * 32 * NCH; i += blockDim.x)
    sk[i / (32 * NCH)][i % (32 * NCH)] = KK(2 * (i / (32 * NCH)) + (i % NCH) / 4);
  asm volatile("fence.proxy.async;" ::: "memory");
  uint32_t *gy = cp + (size_t)warp * 32 * NCH;
  const uint32_t sdb = (uint32_t)__cvta_generic_to_shared(&sd[wb][0]);
  const uint32_t skb = (uint32_t)__cvta_generic_to_shared(&sk[0][0]);
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int s = 0; s < sets; ++s) {
#pragma unroll
    for (int i = 0; i < 8; ++i) imma(d[i], one, one);
    trip<0>(y, z, d, gy, sdb, skb, ln);
    trip<1>(y, z, d, gy, sdb, skb + 4u * 32 * NCH, ln);
    trip<2>(y, z, d, gy, sdb, skb + 8u * 32 * NCH, ln);
    trip<3>(y, z, d, gy, sdb, skb + 12u * 32 * NCH, ln);
  }
  __syncthreads();
  long long t1 = clock64();
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    out[idx * 16 + j] = y[j];
    out[idx * 16 + NCH + j] = z[j];
  }
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

__device__ __forceinline__ void bulk_xor_n(uint32_t *g, uint32_t s, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%1], %2;\n\t"
               "cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group 0;" ::"l"(g), "r"(s), "r"(bytes)
               : "memory");
}

template <int H>
__device__ __forceinline__ void tripb(uint32_t (&y)[NCH], uint32_t (&z)[NCH], const uint32_t (&d)[8][4], uint32_t *gb,
                                      uint32_t sd, uint32_t sk, uint32_t bytes, int tid) {
  uint32_t m[NCH];
#pragma unroll
  for (int j = 0; j < NCH; ++j) m[j] = d[2 * H + j / 4][j % 4] * FK(4 * (2 * H + j / 4) + j % 4);
  uint32_t *g = gb + NCH * tid;
  st128(g, y[0], y[1], y[2], y[3]);
  st128(g + 4, y[4], y[5], y[6], y[7]);
  const uint32_t s = sd + 4u * NCH * tid;
  sts64(s, m[0], m[1]);
  sts64(s + 8, m[2], m[3]);
  sts64(s + 16, m[4], m[5]);
  sts64(s + 24, m[6], m[7]);
  asm volatile("fence.proxy.async;" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    bulk_xor_n(gb, sd, bytes);
    bulk_xor_n(gb, sk, bytes);
    asm volatile("fence.proxy.async;" ::: "memory");
  }
  __syncthreads();
  uint32_t u[NCH];
  ldcg128(g, u[0], u[1], u[2], u[3]);
  ldcg128(g + 4, u[4], u[5], u[6], u[7]);
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    y[j] = u[j] * KA(2 * H + j / 4) + z[j];
    z[j] = __funnelshift_l(u[j], u[j], ROT(2 * H + j / 4)) + y[j];
  }
}

// mode 9; `cp` holds each block's NT NCH-word y slot
template <int MODE>
__global__ void __launch_bounds__(NT) cpwb(const uint32_t *st, const uint32_t *W, uint32_t *out, uint32_t *cp,
                                           long long *cyc, int sets, uint32_t one) {
  static_assert(MODE == 9, "cpwb is mode 9");
  const int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
  const int src = idx & 1023;
  __shared__ __align__(128) uint32_t sd[NT * NCH];
  __shared__ __align__(128) uint32_t sk[4][NT * NCH];
  uint32_t y[NCH], z[NCH], d[8][4];
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    y[j] = st[src * 16 + j];
    z[j] = st[src * 16 + NCH + j];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[i][e] = W[src * 32 + 4 * i + e];
  for (int i = tid; i < 4 * NT * NCH; i += blockDim.x)
    sk[i / (NT * NCH)][i % (NT * NCH)] = KK(2 * (i / (NT * NCH)) + (i % NCH) / 4);
  asm volatile("fence.proxy.async;" ::: "memory");
  uint32_t *gb = cp + (size_t)blockIdx.x * NT * NCH;
  const uint32_t bytes = blockDim.x * NCH * 4u;
  const uint32_t sdb = (uint32_t)__cvta_generic_to_shared(&sd[0]);
  const uint32_t skb = (uint32_t)__cvta_generic_to_shared(&sk[0][0]);
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int s = 0; s < sets; ++s) {
#pragma unroll
    for (int i = 0; i < 8; ++i) imma(d[i], one, one);
    tripb<0>(y, z, d, gb, sdb, skb, bytes, tid);
    tripb<1>(y, z, d, gb, sdb, skb + 4u * NT * NCH, bytes, tid);
    tripb<2>(y, z, d, gb, sdb, skb + 8u * NT * NCH, bytes, tid);
    tripb<3>(y, z, d, gb, sdb, skb + 12u * NT * NCH, bytes, tid);
  }
  __syncthreads();
  long long t1 = clock64();
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    out[idx * 16 + j] = y[j];
    out[idx * 16 + NCH + j] = z[j];
  }
  if (tid == 0) cyc[blockIdx.x] = t1 - t0;
}

// mode 10; `cp` holds each block's BT NCH-word y slot, and the launch brings 5 BT NCH words of shared memory
template <int MODE>
__global__ void __launch_bounds__(4 * NT) cpwx(const uint32_t *st, const uint32_t *W, uint32_t *out, uint32_t *cp,
                                               long long *cyc, int sets, uint32_t one) {
  static_assert(MODE == 10, "cpwx is mode 10");
  constexpr int BT = 4 * NT;
  extern __shared__ __align__(128) uint32_t smem[];
  const int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
  const int src = idx & 1023;
  uint32_t *sd = smem, *sk = smem + BT * NCH;
  uint32_t y[NCH], z[NCH], d[8][4];
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    y[j] = st[src * 16 + j];
    z[j] = st[src * 16 + NCH + j];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[i][e] = W[src * 32 + 4 * i + e];
  for (int i = tid; i < 4 * BT * NCH; i += blockDim.x) sk[i] = KK(2 * (i / (BT * NCH)) + (i % NCH) / 4);
  asm volatile("fence.proxy.async;" ::: "memory");
  uint32_t *gb = cp + (size_t)blockIdx.x * BT * NCH;
  const uint32_t bytes = blockDim.x * NCH * 4u;
  const uint32_t sdb = (uint32_t)__cvta_generic_to_shared(sd);
  const uint32_t skb = (uint32_t)__cvta_generic_to_shared(sk);
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int s = 0; s < sets; ++s) {
#pragma unroll
    for (int i = 0; i < 8; ++i) imma(d[i], one, one);
    tripb<0>(y, z, d, gb, sdb, skb, bytes, tid);
    tripb<1>(y, z, d, gb, sdb, skb + 4u * BT * NCH, bytes, tid);
    tripb<2>(y, z, d, gb, sdb, skb + 8u * BT * NCH, bytes, tid);
    tripb<3>(y, z, d, gb, sdb, skb + 12u * BT * NCH, bytes, tid);
  }
  __syncthreads();
  long long t1 = clock64();
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    out[idx * 16 + j] = y[j];
    out[idx * 16 + NCH + j] = z[j];
  }
  if (tid == 0) cyc[blockIdx.x] = t1 - t0;
}

// the device pass sees no host code, so each mode is instantiated here
#define CPW_INST(M) \
  template __global__ void cpw<M>(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *, int, uint32_t);
CPW_INST(0)
CPW_INST(1)
CPW_INST(2)
CPW_INST(3)
CPW_INST(4)
CPW_INST(5)
CPW_INST(6)
CPW_INST(7)
template __global__ void cpwm<8>(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *, int, uint32_t);
template __global__ void cpwb<9>(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *, int, uint32_t);
template __global__ void cpwx<10>(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *, int, uint32_t);

#ifndef __CUDA_ARCH__
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

typedef void (*Kern)(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *, int, uint32_t);

static uint32_t h_rotl(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

static uint32_t st[1024 * 16], W[1024 * 32];

// n threads through `sets` sets: final states (thread-major, y then z), and each set's checkpoint (y, z) of the
// first warp
static void ref(int sets, int n, uint32_t *fin, uint32_t *ck) {
  uint32_t(*S)[16] = (uint32_t(*)[16])fin;
  memcpy(fin, st, (size_t)n * 16 * sizeof(uint32_t));
  for (int s = 0; s < sets; ++s) {
    uint32_t cy = 0, cz = 0;
    for (int ln = 0; ln < n; ++ln) {
      uint32_t *y = S[ln], *z = S[ln] + NCH;
      for (int t = 0; t < 8; ++t)
        for (int e = 0; e < 4; ++e) {
          int c = 4 * (t % 2) + e;
          uint32_t p = W[ln * 32 + 4 * t + e] + 32u * (uint32_t)(s + 1);
          uint32_t u = y[c] ^ (p * FK(4 * t + e)) ^ KK(t);
          y[c] = u * KA(t) + z[c];
          z[c] = h_rotl(u, ROT(t)) + y[c];
        }
      if (ln < 32)
        for (int j = 0; j < NCH; ++j) {
          cy ^= y[j];
          cz ^= z[j];
        }
    }
    ck[2 * s] = cy;
    ck[2 * s + 1] = cz;
  }
}

int main() {
  int dev = 0, sms = 0, maj = 0, mnr = 0, clk = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev);
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
  printf("device cc=%d.%d sms=%d clock_khz=%d\n", maj, mnr, sms, clk);

  srand(0xC0B1);
  for (int i = 0; i < 1024 * 16; ++i) st[i] = (uint32_t)rand() * 668265263u ^ (uint32_t)rand();
  for (int i = 0; i < 1024 * 32; ++i) W[i] = (uint32_t)rand() * 2654435761u ^ (uint32_t)rand();
  const int BSETS = CPS, TSETS = 2048;
  static uint32_t fin[4 * NT * 16], ck[2 * CPS];
  ref(BSETS, 4 * NT, fin, ck);

  // each mode's block and dynamic shared memory; its bit-exact pass runs one warp, or with `whole` one whole block
  struct { const char *name; Kern k; int mode, nt, smem; bool whole; } ks[] = {
      {"cpw_none", cpw<0>, 0, NT, 0, false},
      {"cpw_st128", cpw<1>, 1, NT, 0, false},
      {"cpw_redux16", cpw<2>, 2, NT, 0, false},
      {"cpw_redg32", cpw<3>, 3, NT, 0, false},
      {"cpw_atoms32", cpw<4>, 4, NT, 0, false},
      {"cpw_redg16", cpw<5>, 5, NT, 0, false},
      {"cpw_atoms16", cpw<6>, 6, NT, 0, false},
      {"cpw_redux2", cpw<7>, 7, NT, 0, false},
      {"cpw_foldmem", cpwm<8>, 8, NT, 0, false},
      {"cpw_foldmem_blk", cpwb<9>, 9, NT, 0, false},
      {"cpw_foldmem_blk16", cpwx<10>, 10, 4 * NT, 5 * 4 * NT * NCH * 4, true},
  };
  const int maxblocks = sms * 32;
  // a warp's 2 CPS checkpoint words, or mode 8's 32 NCH-word y slot
  const size_t nthr = (size_t)maxblocks * NT, ncp = (nthr / 32) * (2 * CPS > 32 * NCH ? 2 * CPS : 32 * NCH);
  uint32_t *d_st, *d_W, *d_out, *d_cp;
  long long *d_cyc;
  cudaMalloc(&d_st, sizeof st);
  cudaMalloc(&d_W, sizeof W);
  cudaMalloc(&d_out, nthr * 16 * sizeof(uint32_t));
  cudaMalloc(&d_cp, ncp * sizeof(uint32_t));
  cudaMalloc(&d_cyc, maxblocks * sizeof(long long));
  cudaMemcpy(d_st, st, sizeof st, cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, W, sizeof W, cudaMemcpyHostToDevice);
  long long *cyc = (long long *)malloc(maxblocks * sizeof(long long));
  const uint32_t one = 0x01010101u;
  int all = 1;
  for (auto &e : ks) {
    static uint32_t out[4 * NT * 16], cpo[2 * CPS];
    if (e.smem > 48 * 1024)
      cudaFuncSetAttribute((const void *)e.k, cudaFuncAttributeMaxDynamicSharedMemorySize, e.smem);
    const int bn = e.whole ? e.nt : 32;
    cudaMemset(d_cp, 0, 2 * CPS * sizeof(uint32_t));
    e.k<<<1, bn, e.smem>>>(d_st, d_W, d_out, d_cp, d_cyc, BSETS, one);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, (size_t)bn * 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpo, d_cp, sizeof cpo, cudaMemcpyDeviceToHost);
    int bad_st = err != cudaSuccess, bad_cp = 0;
    for (int i = 0; i < bn * 16; ++i) bad_st += out[i] != fin[i];
    const bool checked = e.mode >= 2 && e.mode <= 7;
    if (checked)
      for (int i = 0; i < 2 * CPS; ++i) bad_cp += cpo[i] != ck[i];
    int ok = bad_st == 0 && bad_cp == 0;
    all &= ok;
    printf("%s bitexact=%d threads=%d bad_states=%d bad_checkpoints=%d checkpoint=%s err=%s\n", e.name, ok, bn, bad_st,
           bad_cp, checked ? "checked" : "none", cudaGetErrorString(err));
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, e.k, e.nt, e.smem);
    if (occ > 32) occ = 32;
    int bpsm[] = {1, 2, 4, occ};
    for (int b : bpsm) {
      if (b > occ) continue;
      int blocks = sms * b;
      cudaMemset(d_cp, 0, ncp * sizeof(uint32_t));
      e.k<<<blocks, e.nt, e.smem>>>(d_st, d_W, d_out, d_cp, d_cyc, TSETS, one);
      err = cudaGetLastError();
      if (err == cudaSuccess) err = cudaDeviceSynchronize();
      cudaMemcpy(cyc, d_cyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
      long long mx = 0;
      for (int i = 0; i < blocks; ++i) mx = cyc[i] > mx ? cyc[i] : mx;
      double wps = b * (e.nt / 32) / 4.0;
      double tp = (double)mx / (wps * TSETS);
      printf("%s tp blocks_per_sm=%d occ=%d warps_per_smsp=%.1f smsp_cyc_per_warp_set=%.4f smsp_cyc_per_unit=%.5f "
             "err=%s\n", e.name, b, occ, wps, tp, tp / 8.0, cudaGetErrorString(err));
    }
  }
  printf("all_bitexact=%d\n", all);
  return all ? 0 : 3;
}
#endif
