// The served sm_120 unit with its CP-W checkpoint (`PouwWarpCheck_v2` at fold fx1), for what each route
// to the checkpoint costs in SASS and on the device. A unit is one IMMA.16832 (128 words a warp, 4 a lane) and fx1's
// absorb of its words (`segment._fx1_absorb`: d = p fx1_key(q), u = y ^ d ^ KK[t], y = u KA[t] + z, z = rotl(u, ROT[t])
// + y, chain 4 (t % 2) + e); a set is 8 units (32 words a lane), and after every set the warp's checkpoint is two words,
// the XOR over its 32 lanes and 8 chains of y and of z (`segment.warp_checkpoint`). Keys are compile-time constants, as
// in the served kernel. A and B are all-ones s8 fragments and D accumulates, so word q of set s is W[q] + 32 (s + 1)
// whatever the fragment layout, and the host recomputes every lane. Modes, the checkpoint's route:
//   0 none (the floor program's fold, no checkpoint);
//   1 redux: each lane XORs its 16 state words to two (7 LOP3), REDUX.XOR each, lane 0 stores both;
//   2 red64: the lane's two words as one 64-bit RED.XOR into the zeroed receipt word pair;
//   3 atoms64: the same into shared memory (ATOMS.XOR.64), flushed after the last set;
//   4 shfl: a 5-step SHFL butterfly a part instead of REDUX;
//   5 egress: STSM x4 of the lane's 16 state words, the cheapest way out of the registers (no reduction: a lower-bound
//     probe, not a checkpoint);
//   6 pair2: 6 LOP3 take the lane's 16 words to two (y, z) pairs, and two 64-bit red.global.xor put both into the
//     zeroed receipt word pair: 8 instructions a set-warp, the issue-merged bound (`cpw_floor.py`);
//   7 st256: two 256-bit st.global.v8.b32 of the lane's 16 state words, the widest lane store (egress only, like 5;
//     built only where ptxas takes it, -DCPW_ST256);
//   8 pair64: the lane's 8 (y, z) pairs XORed as 64-bit words, then one 64-bit red.global.xor: whether ptxas has a
//     64-bit logic op that merges more than two words a lane an instruction.
// `main` runs on the device it finds: bitexact (one warp, 64 sets, every final state and every checkpoint against the
// host), then tp at 1, 2, 4 warps a SMSP and at full occupancy: SMSP-cycles a warp-set (clock64 a block, max over
// blocks), 8 units a warp-set.
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

__device__ __forceinline__ void red_xor64(unsigned long long *p, uint32_t lo, uint32_t hi) {
  asm volatile("red.relaxed.gpu.global.xor.b64 [%0], %1;" ::"l"(p), "l"(((unsigned long long)hi << 32) | lo)
               : "memory");
}

__device__ __forceinline__ void stsm4(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(a), "r"(b), "r"(c),
               "r"(d)
               : "memory");
}

#ifdef CPW_ST256
__device__ __forceinline__ void st256(uint32_t *p, const uint32_t (&v)[NCH]) {
  asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(p), "r"(v[0]), "r"(v[1]), "r"(v[2]),
               "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7])
               : "memory");
}
#endif

template <int MODE>
__global__ void __launch_bounds__(NT) cpw(const uint32_t *st, const uint32_t *W, uint32_t *out, uint32_t *cp,
                                          long long *cyc, int sets, uint32_t one) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x, ln = idx & 31, warp = idx >> 5, wb = threadIdx.x >> 5;
  const int src = idx & 1023;
  __shared__ unsigned long long scp[NT / 32][CPS];
  __shared__ __align__(16) uint32_t sbuf[NT / 32][32 * 16];
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
  if (MODE == 3)
    for (int k = ln; k < CPS; k += 32) scp[wb][k] = 0;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(&sbuf[wb][0]) + (uint32_t)ln * 16u;
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
    const int k = warp * CPS + (s & (CPS - 1));
    if (MODE >= 1 && MODE <= 4) {
      uint32_t xy = y[0], xz = z[0];
#pragma unroll
      for (int j = 1; j < NCH; ++j) {
        xy ^= y[j];
        xz ^= z[j];
      }
      if (MODE == 1) {
        uint32_t ry = __reduce_xor_sync(0xffffffffu, xy), rz = __reduce_xor_sync(0xffffffffu, xz);
        if (ln == 0) *(uint2 *)&cp[2 * k] = make_uint2(ry, rz);
      } else if (MODE == 2) {
        atomicXor((unsigned long long *)&cp[2 * k], ((unsigned long long)xz << 32) | xy);
      } else if (MODE == 3) {
        atomicXor(&scp[wb][s & (CPS - 1)], ((unsigned long long)xz << 32) | xy);
      } else {
#pragma unroll
        for (int o = 16; o; o >>= 1) {
          xy ^= __shfl_xor_sync(0xffffffffu, xy, o);
          xz ^= __shfl_xor_sync(0xffffffffu, xz, o);
        }
        if (ln == 0) *(uint2 *)&cp[2 * k] = make_uint2(xy, xz);
      }
    } else if (MODE == 6) {
      const uint32_t ya = y[0] ^ y[1] ^ y[2] ^ y[6] ^ y[7], yb = y[3] ^ y[4] ^ y[5];
      const uint32_t za = z[0] ^ z[1] ^ z[2] ^ z[6] ^ z[7], zb = z[3] ^ z[4] ^ z[5];
      unsigned long long *p = (unsigned long long *)&cp[2 * k];
      red_xor64(p, ya, za);
      red_xor64(p, yb, zb);
    } else if (MODE == 7) {
#ifdef CPW_ST256
      st256(&out[idx * 16], y);
      st256(&out[idx * 16 + NCH], z);
#endif
    } else if (MODE == 8) {
      unsigned long long v = ((unsigned long long)z[0] << 32) | y[0];
#pragma unroll
      for (int j = 1; j < NCH; ++j) v ^= ((unsigned long long)z[j] << 32) | y[j];
      asm volatile("red.relaxed.gpu.global.xor.b64 [%0], %1;" ::"l"((unsigned long long *)&cp[2 * k]), "l"(v)
                   : "memory");
    } else if (MODE == 5) {
      stsm4(sb, y[0], y[1], y[2], y[3]);
      stsm4(sb + 512, y[4], y[5], y[6], y[7]);
      stsm4(sb + 1024, z[0], z[1], z[2], z[3]);
      stsm4(sb + 1536, z[4], z[5], z[6], z[7]);
    }
  }
  __syncthreads();
  long long t1 = clock64();
  if (MODE != 7) {
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      out[idx * 16 + j] = y[j];
      out[idx * 16 + NCH + j] = z[j];
    }
  }
  if (MODE == 3)
    for (int k = ln; k < CPS; k += 32) ((unsigned long long *)cp)[warp * CPS + k] = scp[wb][k];
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
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
#ifdef CPW_ST256
CPW_INST(7)
#endif
CPW_INST(8)

#ifndef __CUDA_ARCH__
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

typedef void (*Kern)(const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, long long *, int, uint32_t);

static uint32_t h_rotl(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

static uint32_t st[1024 * 16], W[1024 * 32];

// one warp's lanes through `sets` sets: final states (lane-major, y then z) and each set's checkpoint (y, z)
static void ref(int sets, uint32_t *fin, uint32_t *ck) {
  uint32_t S[32][16];
  memcpy(S, st, sizeof S);
  for (int s = 0; s < sets; ++s) {
    uint32_t cy = 0, cz = 0;
    for (int ln = 0; ln < 32; ++ln) {
      uint32_t *y = S[ln], *z = S[ln] + NCH;
      for (int t = 0; t < 8; ++t)
        for (int e = 0; e < 4; ++e) {
          int c = 4 * (t % 2) + e;
          uint32_t p = W[ln * 32 + 4 * t + e] + 32u * (uint32_t)(s + 1);
          uint32_t u = y[c] ^ (p * FK(4 * t + e)) ^ KK(t);
          y[c] = u * KA(t) + z[c];
          z[c] = h_rotl(u, ROT(t)) + y[c];
        }
      for (int j = 0; j < NCH; ++j) {
        cy ^= y[j];
        cz ^= z[j];
      }
    }
    ck[2 * s] = cy;
    ck[2 * s + 1] = cz;
  }
  memcpy(fin, S, sizeof S);
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
  static uint32_t fin[32 * 16], ck[2 * CPS];
  ref(BSETS, fin, ck);

  struct { const char *name; Kern k; int mode; } ks[] = {
      {"cpw_none", cpw<0>, 0}, {"cpw_redux", cpw<1>, 1}, {"cpw_red64", cpw<2>, 2},
      {"cpw_atoms64", cpw<3>, 3}, {"cpw_shfl", cpw<4>, 4}, {"cpw_egress", cpw<5>, 5},
      {"cpw_pair2", cpw<6>, 6},
#ifdef CPW_ST256
      {"cpw_st256", cpw<7>, 7},
#endif
      {"cpw_pair64", cpw<8>, 8},
  };
  const int maxblocks = sms * 32;
  const size_t nthr = (size_t)maxblocks * NT, ncp = (nthr / 32) * CPS * 2;
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
    static uint32_t out[32 * 16], cpo[2 * CPS];
    cudaMemset(d_cp, 0, 2 * CPS * sizeof(uint32_t));
    e.k<<<1, 32>>>(d_st, d_W, d_out, d_cp, d_cyc, BSETS, one);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpo, d_cp, sizeof cpo, cudaMemcpyDeviceToHost);
    int bad_st = err != cudaSuccess, bad_cp = 0;
    for (int i = 0; i < 32 * 16; ++i) bad_st += out[i] != fin[i];
    const bool checked = (e.mode >= 1 && e.mode <= 4) || e.mode == 6 || e.mode == 8;
    if (checked)
      for (int i = 0; i < 2 * CPS; ++i) bad_cp += cpo[i] != ck[i];
    int ok = bad_st == 0 && bad_cp == 0;
    all &= ok;
    printf("%s bitexact=%d bad_states=%d bad_checkpoints=%d checkpoint=%s err=%s\n", e.name, ok, bad_st, bad_cp,
           checked ? "checked" : "none", cudaGetErrorString(err));
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, e.k, NT, 0);
    if (occ > 32) occ = 32;
    int bpsm[] = {1, 2, 4, occ};
    for (int b : bpsm) {
      if (b > occ) continue;
      int blocks = sms * b;
      cudaMemset(d_cp, 0, ncp * sizeof(uint32_t));
      e.k<<<blocks, NT>>>(d_st, d_W, d_out, d_cp, d_cyc, TSETS, one);
      err = cudaGetLastError();
      if (err == cudaSuccess) err = cudaDeviceSynchronize();
      cudaMemcpy(cyc, d_cyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
      long long mx = 0;
      for (int i = 0; i < blocks; ++i) mx = cyc[i] > mx ? cyc[i] : mx;
      double wps = b * (NT / 32) / 4.0;
      double tp = (double)mx / (wps * TSETS);
      printf("%s tp blocks_per_sm=%d occ=%d warps_per_smsp=%.1f smsp_cyc_per_warp_set=%.4f smsp_cyc_per_unit=%.5f "
             "err=%s\n", e.name, b, occ, wps, tp, tp / 8.0, cudaGetErrorString(err));
    }
  }
  printf("all_bitexact=%d\n", all);
  return all ? 0 : 3;
}
#endif
