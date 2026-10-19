// Dither's served linear in one stream (CA's H1V3, 07:04Z): each call's y = dequant(O), O = O' - sum_d 128^d D_d, from the
// noised mainloop that also computes the call's H1 + fx1 receipts (`h1r_unit`'s MMAs over X = A + b and W' = B + c + U and
// its tail, `segment.call_words`), so the MMAs whose words are receipted are the MMAs that serve: no other product of A
// and B is computed.  z = O (s32, mod 2^32) is the z each cell commits.  The seeds are the call's own, from the window's
// salt and root(A_j), root(B_j) (`h1r_seeds`, the 17 Oct seeded regime), so a call needs its A table's root before its
// units and nothing from a later round.  A served call is, on the caller's stream, plain launches into caller-owned
// buffers, the step read from the device (`H1OStep`), so a CUDA graph captured once serves every step:
//   h1o_slot     a warp a slot row: the call's A slot rows < m (xq's bytes), rows >= m zero, and their scales (0 above m)
//   (h1c)        the A table of the slot, its root where the call's seeds read it (`h1c_build` on that one table)
//   h1r_seeds    the call's units' seed_j and σ_R
//   h1o_units    a warp a unit (`h1r_unit<2, OutEpi>`): the receipt, then z and y of the unit's cells; A rows >= m read as
//                the canonical empty row (zero); a dead unit (its row block holds no admitted row) writes zero receipts
//                and zero z and y
// `h1o_ref` is the gate's reference for z, another computation of the same definition: one warp a unit, its noise from
// full compressions (`nblock`, not the served kernel's midstate), T_last and the residue sums Y by scalar dp4a (no MMA),
// and z = T_last - U Y (the digits' recombination is exact: sum_d 128^d A_d = Y).
#include "h1r.cu"
#include <cuda_bf16.h>

struct H1O {
  int8_t* slot[2];      // A slots [T, K] s8
  float* scale[2];      // s_a slots [T]
  int* z[2];            // z slots [T, N] s32
  const H1Call* rc[2];  // the call's entry in parity p's call table
  u32* out[2];          // parity p's receipts (the table's out0 indexes them)
  u64* seeds;           // the units' seeds (seed0 indexes them)
  __nv_bfloat16* ydump; // [T, N] or null: y rows < m copied here under F_DUMP (the gate's view of the served y)
  int t, n, k, units;
};

// flags: F_DUMP copies y rows < m to ydump; F_NW_TILES (no A slot, scale or z write) and F_NW_UNITS (no receipt write) are
// the gate's no-write control, y still written; F_FAULT adds 2^FAULT_SHIFT to unit 0's accumulator of cell (0, 0) after
// segment 0, before that segment's words are absorbed (the conformance test: the receipt and the served z and y must move)
enum { F_DUMP = 1, F_NW_TILES = 2, F_NW_UNITS = 4, F_FAULT = 8 };
constexpr int FAULT_SHIFT = 24;

struct H1OStep {
  int parity, m, pad, flags;
};

struct OutEpi {
  static constexpr bool ON = true;
  int* z;
  __nv_bfloat16* y;
  __nv_bfloat16* ydump;
  const float* sa;
  const float* sw;
  int n, M, m, flags;

  __device__ bool a_ok(int row) const { return row < m; }
  __device__ void dead(int ri, int cj, int lane) const {
    const int row = 32 * ri + lane;
    if (!(flags & F_NW_TILES)) {
      uint4* zr = reinterpret_cast<uint4*>(z + (long long)row * n + 32 * cj);
#pragma unroll
      for (int i = 0; i < 8; ++i) zr[i] = make_uint4(0, 0, 0, 0);
    }
    if (row < M) {
      uint4* yr = reinterpret_cast<uint4*>(y + (long long)row * n + 32 * cj);
#pragma unroll
      for (int i = 0; i < 4; ++i) yr[i] = make_uint4(0, 0, 0, 0);
    }
  }
  __device__ void fault(int u, int lane, int s, int (&acc)[2][4][4]) const {
    if ((flags & F_FAULT) && u == 0 && lane == 0 && s == 0) acc[0][0][0] = (int)((u32)acc[0][0][0] + (1u << FAULT_SHIFT));
  }
  __device__ void out(int ri, int cj, int lane, const int (&acc)[2][4][4]) const {
    const int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const int row = 32 * ri + 16 * mi + g + 8 * eh;
        const float s_a = row < m ? sa[row] : 0.0f;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const int col = 32 * cj + 8 * ni + 2 * q;
          const int z0 = acc[mi][ni][2 * eh], z1 = acc[mi][ni][2 * eh + 1];
          if (!(flags & F_NW_TILES)) *reinterpret_cast<int2*>(z + (long long)row * n + col) = make_int2(z0, z1);
          if (row < M) {
            __nv_bfloat162 yv;
            yv.x = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z0), s_a), sw[col]));
            yv.y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z1), s_a), sw[col + 1]));
            *reinterpret_cast<__nv_bfloat162*>(y + (long long)row * n + col) = yv;
            if ((flags & F_DUMP) && row < m && ydump) *reinterpret_cast<__nv_bfloat162*>(ydump + (long long)row * n + col) = yv;
          }
        }
      }
  }
};

__global__ void h1o_slot_k(const H1O* __restrict__ F, const H1OStep* __restrict__ S, const int8_t* __restrict__ xq,
                           const float* __restrict__ sa) {
  const H1OStep st = *S;
  if (st.flags & F_NW_TILES) return;
  const int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int T = F->t, K = F->k, p = st.parity;
  if (row >= T) return;
  const bool alive = row < st.m;
  uint4* dst = reinterpret_cast<uint4*>(F->slot[p] + (long long)row * K);
  const uint4* src = reinterpret_cast<const uint4*>(xq + (long long)(alive ? row : 0) * K);
  for (int i = lane; i < K / 16; i += 32) dst[i] = alive ? src[i] : make_uint4(0, 0, 0, 0);
  if (lane == 0) F->scale[p][row] = alive ? sa[row] : 0.0f;
}

template <int NW>
__global__ void __launch_bounds__(32 * NW, 3) h1o_units_k(const H1O* __restrict__ F, const H1OStep* __restrict__ S,
                                                          const int8_t* __restrict__ xq, const float* __restrict__ sa, int M,
                                                          const float* __restrict__ sw, __nv_bfloat16* __restrict__ y,
                                                          u64 nt0, u64 nt1, u64 nt2) {
  __shared__ __align__(16) uint8_t smem[NW][2][32 * XS];
  __shared__ u64 sd[NW][24];
  __shared__ u32 nzs[NW][2][32 * NZ];
  const H1OStep st = *S;
  const int p = st.parity;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  H1Call C = *F->rc[p];
  const int v = blockIdx.x * NW + wid;
  if (v >= C.units) return;
  C.a = xq;
  C.lda = F->k;
  const OutEpi ep{F->z[p], y, F->ydump, sa, sw, F->n, M, st.m, st.flags};
  h1r_unit<2, OutEpi>(C, v, lane, smem[wid][0], smem[wid][1], sd[wid], nzs[wid][0], F->seeds, nt0, nt1, nt2, F->out[p],
                      (st.flags & F_NW_UNITS) ? 1 : 0, ep);
}

// the reference: unit u = ri * ncb + cj of call C (A rows from C.a, rows >= the slot's m already zero), z_ref [T, N]
__global__ void __launch_bounds__(32) h1o_ref_k(const H1Call* __restrict__ call, const u64* __restrict__ seeds, u64 nt0,
                                                u64 nt1, u64 nt2, int* __restrict__ zref, int n) {
  __shared__ __align__(16) uint8_t Xs[32][516];
  __shared__ __align__(16) uint8_t Ws[32][516];
  __shared__ int8_t Us[32][32];
  const H1Call C = *call;
  const int u = blockIdx.x, lane = threadIdx.x;
  if (u >= C.units) return;
  const int ri = u / C.ncb, cj = u % C.ncb;
  int* zr = zref + (long long)(32 * ri + lane) * n + 32 * cj;
  if (!C.live[ri]) {
    for (int j = 0; j < 32; ++j) zr[j] = 0;
    return;
  }
  const u64* sj = seeds + (long long)(C.seed0 + u) * 16;
  const int KS = C.k >> 5, nb = KS / 16, RB = nb + 1;
  const u64 rowb = (u64)lane * RB, colb = (u64)32 * RB + (u64)lane * RB;
  {
    u32 h[16];
    nblock(sj, colb + nb, nt0, nt1, nt2, h);
    for (int r = 0; r < 32; ++r) {
      const u32 by = (h[r / 4] >> (8 * (r % 4))) & 0xFFu;
      Us[lane][r] = (int8_t)(by >= 129u ? (int)by - 193 : (int)by - 64);
    }
  }
  int acc[32], Y[32];
  for (int j = 0; j < 32; ++j) acc[j] = Y[j] = 0;
  const int8_t* arow = C.a + (32LL * ri + lane) * C.lda;
  const int8_t* brow = C.b + (32LL * cj + lane) * C.ldb;
  for (int ch = 0; ch < nb; ++ch) {
    u32 hr[16], hc[16];
    nblock(sj, rowb + ch, nt0, nt1, nt2, hr);
    nblock(sj, colb + ch, nt0, nt1, nt2, hc);
    __syncwarp();
    for (int kk = 0; kk < 512; ++kk) {
      const int s16 = kk / 32, t = kk % 32;
      const int a = arow[512 * ch + kk], b = brow[512 * ch + kk];
      Xs[lane][kk] = (uint8_t)(a + (int)((hr[s16] >> t) & 1u));
      Ws[lane][kk] = (uint8_t)(b + (int)((hc[s16] >> t) & 1u) + Us[lane][t]);
    }
    __syncwarp();
    for (int k8 = 0; k8 < 16; ++k8)
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        const int k4 = 8 * k8 + w;
        const int xw = *reinterpret_cast<const int*>(&Xs[lane][4 * k4]);
#pragma unroll
        for (int i = 0; i < 4; ++i) Y[4 * w + i] += (int)(int8_t)(xw >> (8 * i));
#pragma unroll
        for (int j = 0; j < 32; ++j) acc[j] = __dp4a(xw, *reinterpret_cast<const int*>(&Ws[j][4 * k4]), acc[j]);
      }
  }
  __syncwarp();
  for (int j = 0; j < 32; ++j) {
    u32 o = (u32)acc[j];
    for (int r = 0; r < 32; ++r) o -= (u32)((int)Us[j][r] * Y[r]);
    zr[j] = (int)o;
  }
}

constexpr int NWO = 4;

// one served call: xq [M, K] s8 and sa [M] f32 (row_quant's), sw [N] f32 (the layer's), y [M, N] bf16 (the call's output);
// F (the call's entry) and S on the device.  The weights are the call table's.  The A table and the seeds run between
// the two launches: h1o_slot_launch, then the caller's h1c_build and h1r_launch(version 0), then h1o_units_launch.
extern "C" int h1o_slot_launch(const H1O* F, const H1OStep* S, int t, const int8_t* xq, const float* sa, cudaStream_t stream) {
  if (t % 32) return 1;
  h1o_slot_k<<<(unsigned)(t / 4), 128, 0, stream>>>(F, S, xq, sa);
  return (int)cudaGetLastError();
}

extern "C" int h1o_units_launch(const H1O* F, const H1OStep* S, int units, int n, int k, const int8_t* xq, const float* sa,
                                int M, const float* sw, void* y, cudaStream_t stream) {
  if (M <= 0 || n % 32 || k % 512 || k < 512 || units <= 0) return 1;
  u64 nt[3];
  tag3("verity/pouw/noise/v1", nt);
  h1o_units_k<NWO><<<(unsigned)((units + NWO - 1) / NWO), 32 * NWO, 0, stream>>>(F, S, xq, sa, M, sw, (__nv_bfloat16*)y,
                                                                                nt[0], nt[1], nt[2]);
  return (int)cudaGetLastError();
}

extern "C" int h1o_ref_launch(const H1Call* call_dev, int units, const u64* seeds, int* zref, int n, cudaStream_t stream) {
  u64 nt[3];
  tag3("verity/pouw/noise/v1", nt);
  h1o_ref_k<<<(unsigned)units, 32, 0, stream>>>(call_dev, seeds, nt[0], nt[1], nt[2], zref, n);
  return (int)cudaGetLastError();
}

extern "C" int h1o_struct_bytes(int which) { return which == 0 ? (int)sizeof(H1O) : (int)sizeof(H1OStep); }
extern "C" int h1o_fault_shift() { return FAULT_SHIFT; }
