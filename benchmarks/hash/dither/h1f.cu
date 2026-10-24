// Dither's served linear as one GEMM: the int8-exact linear's y = RN_bf16(RN32(RN32(f32(z) s_a) s_w)) from z = xq W^T,
// the H1 tile's exact product (`h1tile`'s z, `int8_exact.int_mm`'s Y), in the epilogue of the kernel that also computes
// the previous step's H1 + fx1 receipts (`h1r_unit`, gated bit for bit against receipts.py).  One plain launch a linear call
// on the caller's stream into caller-owned buffers, the step read from the device (`H1FStep`), so a CUDA graph captured once
// serves every step.
//
// A step s's receipts need its coin, and its coin needs root(A_s) and root(z_s) committed first, so step s's units are
// computed during step s + 1's call of the same linear (lag one): parity p = s mod 2 picks the slots a call writes, p ^ 1
// the slots its receipts read.  Blocks of one launch, in order:
//   z tiles   one block a 32 x 32 tile of z = xq W^T over rows [0, roundup32(M)), its NW warps each a quarter of K, their
//             s32 partial sums added in shared memory (exact): y for rows < M, z slot rows < m (rows >= m zero), A slot
//             rows < m (xq's bytes; rows >= m zero) and their scales (0 for an empty slot)
//   clear     a warp a slot row of [roundup32(M), T) zeroed (z, A and scale), so every empty slot is the canonical empty row
//   units     a warp a unit of the previous step's units of this call (`H1F.rc[p ^ 1]`), when `H1FStep.rc_on`; a launch with
//             units = 0 runs the instantiation without them (the receipts computed elsewhere: `h1r_launch` on another
//             stream)
#include "h1r.cu"
#include <cuda_bf16.h>

struct H1F {
  int8_t* slot[2];      // A slots [T, K] s8
  float* scale[2];      // s_a slots [T]
  int* z[2];            // z slots [T, N] s32
  const H1Call* rc[2];  // the receipts' call reading parity p's slots
  __nv_bfloat16* ydump; // [T, N] or null: y rows < m copied here under F_DUMP (the gate's view of the served y)
  int t, n, k, pad;
};

// flags: F_DUMP copies y rows < m to ydump; F_NW_TILES and F_NW_UNITS are the gate's no-write control (no z, A slot or
// scale write and no slot clear; no receipt write), y still written, so serving goes on while the transcript is not
enum { F_DUMP = 1, F_NW_TILES = 2, F_NW_UNITS = 4 };

struct H1FStep {
  int parity, m, rc_on, flags;
};

template <int NW, bool UNITS>
__global__ void __launch_bounds__(32 * NW, UNITS ? 3 : 4) h1f_k(const H1F* __restrict__ F, const H1FStep* __restrict__ S,
                                                                const int8_t* __restrict__ xq, const float* __restrict__ sa,
                                                                int M, const int8_t* __restrict__ w,
                                                                const float* __restrict__ sw, __nv_bfloat16* __restrict__ y,
                                                                const u64* __restrict__ seeds, u64 nt0, u64 nt1, u64 nt2,
                                                                u32* __restrict__ out) {
  static_assert(NW == 4, "the tile's partial sums fill smem exactly at NW = 4");
  __shared__ __align__(16) uint8_t smem[NW][2][32 * XS];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const H1FStep st = *S;
  const int p = st.parity, m = st.m;
  const bool nwt = st.flags & F_NW_TILES;
  const int T = F->t, N = F->n, K = F->k, ncb = N >> 5, KS = K >> 5;
  const int Mz = (M + 31) & ~31;
  const int nrz = Mz >> 5, nzt = nrz * ncb, ncl = T - Mz;
  uint8_t* Xs = smem[wid][0];
  uint8_t* Ws = smem[wid][1];
  if ((int)blockIdx.x < nzt) {
    const int v = blockIdx.x;
    const int ri = v % nrz, cj = v / nrz;
    const int per = KS / NW, s0 = wid * per, s1 = s0 + per;  // 512 | K, so NW | KS
    const int g = lane >> 2, q = lane & 3;
    const int ar = 32 * ri + lane;
    const bool aok = ar < M, alive = ar < m;
    const int8_t* arow = xq + (long long)(aok ? ar : 0) * K;
    const int8_t* brow = w + (long long)(32 * cj + lane) * K;
    int8_t* srow = F->slot[p] + (long long)ar * K;
    int acc[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mi][ni][e] = 0;
    const uint4 zero4 = make_uint4(0, 0, 0, 0);
    uint4 pa0 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * s0) : zero4;
    uint4 pa1 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * s0 + 16) : zero4;
    uint4 pb0 = *reinterpret_cast<const uint4*>(brow + 32 * s0);
    uint4 pb1 = *reinterpret_cast<const uint4*>(brow + 32 * s0 + 16);
    for (int s = s0; s < s1; ++s) {
      const uint4 a0 = pa0, a1 = pa1, b0 = pb0, b1 = pb1;
      if (s + 1 < s1) {
        pa0 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * (s + 1)) : zero4;
        pa1 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * (s + 1) + 16) : zero4;
        pb0 = *reinterpret_cast<const uint4*>(brow + 32 * (s + 1));
        pb1 = *reinterpret_cast<const uint4*>(brow + 32 * (s + 1) + 16);
      }
      if (!nwt && s % ncb == cj) {
        *reinterpret_cast<uint4*>(srow + 32 * s) = alive ? a0 : zero4;
        *reinterpret_cast<uint4*>(srow + 32 * s + 16) = alive ? a1 : zero4;
      }
      *reinterpret_cast<uint4*>(Xs + lane * XS) = a0;
      *reinterpret_cast<uint4*>(Xs + lane * XS + 16) = a1;
      *reinterpret_cast<uint4*>(Ws + lane * XS) = b0;
      *reinterpret_cast<uint4*>(Ws + lane * XS + 16) = b1;
      __syncwarp();
      u32 af[2][4], bf[4][2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        af[mi][0] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 4 * q);
        af[mi][1] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 4 * q);
        af[mi][2] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 16 + 4 * q);
        af[mi][3] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 16 + 4 * q);
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        bf[ni][0] = *reinterpret_cast<const u32*>(Ws + (8 * ni + g) * XS + 4 * q);
        bf[ni][1] = *reinterpret_cast<const u32*>(Ws + (8 * ni + g) * XS + 16 + 4 * q);
      }
      __syncwarp();
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) mma(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
    }
    __syncthreads();
    int* red = reinterpret_cast<int*>(&smem[0][0][0]);  // (NW - 1) x 32 lanes x 32 partial sums, 12 KiB
    if (wid > 0) {
#pragma unroll
      for (int j = 0; j < 32; ++j) red[(wid - 1) * 1024 + 32 * j + lane] = acc[j >> 4][(j >> 2) & 3][j & 3];
    }
    __syncthreads();
    if (wid > 0) return;
#pragma unroll
    for (int o = 0; o < NW - 1; ++o)
#pragma unroll
      for (int j = 0; j < 32; ++j) acc[j >> 4][(j >> 2) & 3][j & 3] += red[o * 1024 + 32 * j + lane];
    if (cj == 0 && !nwt) F->scale[p][ar] = alive ? sa[ar] : 0.0f;
    int* zs = F->z[p];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const int row = 32 * ri + 16 * mi + g + 8 * eh;
        const bool rok = row < M, rlive = row < m;
        const float s_a = rok ? sa[row] : 0.0f;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const int col = 32 * cj + 8 * ni + 2 * q;
          const int z0 = acc[mi][ni][2 * eh], z1 = acc[mi][ni][2 * eh + 1];
          if (!nwt)
            *reinterpret_cast<int2*>(zs + (long long)row * N + col) = rlive ? make_int2(z0, z1) : make_int2(0, 0);
          if (rok) {
            __nv_bfloat162 yv;
            yv.x = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z0), s_a), sw[col]));
            yv.y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z1), s_a), sw[col + 1]));
            *reinterpret_cast<__nv_bfloat162*>(y + (long long)row * N + col) = yv;
            if ((st.flags & F_DUMP) && rlive && F->ydump)
              *reinterpret_cast<__nv_bfloat162*>(F->ydump + (long long)row * N + col) = yv;
          }
        }
      }
    return;
  }
  const int v = ((int)blockIdx.x - nzt) * NW + wid;
  if (v < ncl) {
    if (nwt) return;
    const int row = Mz + v;
    uint4* zr = reinterpret_cast<uint4*>(F->z[p] + (long long)row * N);
    for (int i = lane; i < N / 4; i += 32) zr[i] = make_uint4(0, 0, 0, 0);
    uint4* ar = reinterpret_cast<uint4*>(F->slot[p] + (long long)row * K);
    for (int i = lane; i < K / 16; i += 32) ar[i] = make_uint4(0, 0, 0, 0);
    if (lane == 0) F->scale[p][row] = 0.0f;
    return;
  }
  if constexpr (UNITS) {
    __shared__ u64 sd[NW][24];
    __shared__ u32 nzs[NW][2][32 * NZ];
    if (!st.rc_on) return;
    const H1Call C = *F->rc[p ^ 1];
    const int u = v - ncl;
    if (u >= C.units) return;
    h1r_unit<2>(C, u, lane, Xs, Ws, sd[wid], nzs[wid][0], seeds, nt0, nt1, nt2, out, (st.flags & F_NW_UNITS) ? 1 : 0);
  }
}

// one served linear call: xq [M, K] s8 and sa [M] f32 (row_quant's), w [N, K] s8 and sw [N] f32 (the layer's), y [M, N]
// bf16 (the call's output); F, S, seeds and out the window's (device).  units: the receipts' call's unit count, or 0 when
// this launch computes no receipts.
extern "C" int h1f_launch(const H1F* F, const H1FStep* S, int t, int n, int k, int units, const int8_t* xq, const float* sa,
                          int M, const int8_t* w, const float* sw, void* y, const u64* seeds, u32* out, cudaStream_t stream) {
  if (M <= 0 || M > t || n % 32 || k % 512 || t % 32 || units < 0) return 1;
  u64 nt[3];
  tag3("verity/pouw/noise/v1", nt);
  const int Mz = (M + 31) & ~31;
  const long long blocks = (long long)(Mz / 32) * (n / 32) + ((t - Mz) + units + NW - 1) / NW;
  if (units)
    h1f_k<NW, true><<<(unsigned)blocks, 32 * NW, 0, stream>>>(F, S, xq, sa, M, w, sw, (__nv_bfloat16*)y, seeds, nt[0], nt[1],
                                                              nt[2], out);
  else
    h1f_k<NW, false><<<(unsigned)blocks, 32 * NW, 0, stream>>>(F, S, xq, sa, M, w, sw, (__nv_bfloat16*)y, seeds, nt[0], nt[1],
                                                               nt[2], out);
  return (int)cudaGetLastError();
}

extern "C" int h1f_struct_bytes(int which) { return which == 0 ? (int)sizeof(H1F) : (int)sizeof(H1FStep); }
