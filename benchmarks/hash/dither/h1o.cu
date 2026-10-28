// Dither's served linear in one stream (H1V3, with its exact wrap correction): each call's
// y = dequant(z), z = O' - sum_d 128^d D_d + 256 X (wb⁺ - wb⁻)^T, from the noised mainloop that also computes the call's
// H1 + fx1 receipts (`h1r_unit`'s MMAs over X = A + b and W' = B + c + U and its tail, `segment.call_words`), so the MMAs
// whose words are receipted are the MMAs that serve: no other product of A and B is computed.  wb⁺ = [B + c + U > 127] and
// wb⁻ = [B + c + U < -128] come from B and the noise in the unit (never an input), so z = X (B + c)^T exactly for every
// int8 B (s32, mod 2^32), the z each cell commits.  The seeds are the call's own, from the window's
// salt and root(A_j), root(B_j) (`unit_seed`, the seeded regime), so a call needs its A table's root before its
// units and nothing from a later round.  A served call is two plain launches on the caller's stream into caller-owned
// buffers, the step read from the device (`H1OStep`), so a CUDA graph captured once serves every step:
//   (h1c_a_call) the call's A slot rows (rows < m xq's bytes and scales, rows >= m the canonical empty row) and their
//                table, its root where the units' seeds read it (h1c.cu; its row-independent parts once a step)
//   h1o_units    a warp a unit (`h1r_unit<2, OutEpi>`): its seed_j and σ_R, the receipt, then z and y of the unit's cells;
//                A rows >= m read as the canonical empty row (zero); a dead unit (its row tile holds no admitted row)
//                writes zero receipts and zero z and y.  `h1o_unitsb` is the same in the block form (`h1b_unit<OutEpi>`,
//                a unit a block of five warps), for decode, where a call's units would leave most SMs idle, and
//                `h1o_unitss` in the split form (`h1s_unit<OutEpi>`, six warps: the hashing a chunk ahead of the
//                building, the fold in two halves).  `h1o_unitsbp` and `h1o_unitssp` are those two peeled
//                (H1V3's PouwPeelSeg): z = O + 256 X (wb⁺ - wb⁻)ᵀ - X cᵀ - (b - 256 xa) Bᵀ = A Bᵀ, today's
//                product from the same O, so y = dequant(A Bᵀ) serves the registered model; the receipts are the same
//                words, and the peel adds binary-by-int8 MMAs beside the wrap MMA (no product of A itself)
//   h1o_units{s,b}2[p]  those kernels as separate functions whose shared-memory carveout `h1o_init` sets to the maximum
//                (102.4 KB on sm_120; the driver's default for the others is 64 KB, one block an SM), and in the block
//                form with every barrier id an immediate (`h1b_unit<E, true>`: 11 barriers, not the 16 a register id
//                reserves, which also caps an SM at one block): the same values; the split form needs its 16
// `h1o_ref` is the gate's reference for z, another computation of the same definition: one warp a unit, its noise from
// full compressions (`nblock`, not the served kernel's midstate), T_last and the residue sums Y by scalar dp4a (no MMA),
// and z = T_last - U Y (the digits' recombination is exact: sum_d 128^d A_d = Y) + 256 sum over the entries whose integer
// B + c + U leaves [-128, 127] of ±X (a scalar scatter, not the served kernel's byte masks and MMA); it counts them, wb's
// density, and the entries with A = 127 and b = 1 (xa, the activation wrap the peel corrects).
#include "h1r.cu"
#include <cuda_bf16.h>
#include <time.h>

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
// segment 0, before that segment's words are absorbed (the conformance test, peeled or not: the receipt and the served z
// and y must move, z in that cell alone); F_FAULT_WB flips unit 0's wb⁺ bit at column 0, entry 0 (only column 0's z and y
// may move, by 256 (1 - 2 wb⁺) X[:, 0]); F_FAULT_XA, peeled, flips unit 0's xa bit at row 0, entry 0 (only row 0's z and
// y in unit 0's columns may move, by 256 (1 - 2 xa) B[:, 0]; neither wb nor xa is receipted)
enum { F_DUMP = 1, F_NW_TILES = 2, F_NW_UNITS = 4, F_FAULT = 8, F_FAULT_WB = 16, F_FAULT_XA = 32 };
constexpr int FAULT_SHIFT = 24;

struct H1OStep {
  int parity, m, pad, flags;
};

template <bool PL, int BD = 0>
struct OutEpiT {
  static constexpr bool ON = true;
  static constexpr bool PEEL = PL;  // z = A Bᵀ from O (h1r.cu's E::PEEL)
  static constexpr bool SEEDS = true;
  static constexpr int BAND = BD;  // J10 banded (h1o_band_launch; NoEpi::BAND): the rows from bws, band leader (ri, (cj / g) g)'s
  int* z;
  int* zo;  // [M, N] s32: the op's z output (rows < M of the slot's z), what the credit gate reads beside y
  __nv_bfloat16* y;
  __nv_bfloat16* ydump;
  const float* sa;
  const float* sw;
  int n, M, m, flags;
  uint8_t* wbs;  // the warp's wb operand, 32 * XS bytes of shared memory
  const u64* salt;       // the window's salt (8 words)
  const u64* step_base;  // the step's first position
  u64 st0, st1, st2, rt0, rt1, rt2;  // the seed and receipt tags
  const u32* bws = nullptr;  // BAND: the call's band workspace (h1o_bandrows_k), band (ri, b) at 16 (ri nl + b) 32 RB words
  int g = 1;                 // BAND: column tiles a band

  __device__ const u32* band_rows(const H1Call& C, int ri, int cj, int RB) const {
    const int nl = (C.ncb + g - 1) / g;
    return bws + 16ull * ((unsigned long long)(ri * nl + cj / g) * 32 * RB);
  }
  __device__ bool a_ok(int row) const { return row < m; }
  __device__ void seed(const H1Call& C, int u, int kind, u64* o) const {
    u64 s[8];
    unit_seed(C, C.base + *step_base + u, salt, kind ? rt0 : st0, kind ? rt1 : st1, kind ? rt2 : st2, s);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = s[j];
  }
  __device__ void wb_fault(int u, int lane, int s, u32 (&P)[8]) const {
    if ((flags & F_FAULT_WB) && u == 0 && lane == 0 && s == 0) P[0] ^= 0x80u;
  }
  // xa's fragment (row half mi); lane 0's word 0 of mi = 0 is row 0, entries 0 .. 3
  __device__ void xa_fault(int u, int lane, int s, int mi, u32 (&xa)[4]) const {
    if ((flags & F_FAULT_XA) && u == 0 && lane == 0 && s == 0 && mi == 0) xa[0] ^= 1u;
  }
  __device__ void dead(int ri, int cj, int lane) const {
    const int row = 32 * ri + lane;
    if (!(flags & F_NW_TILES)) {
      uint4* zr = reinterpret_cast<uint4*>(z + (long long)row * n + 32 * cj);
#pragma unroll
      for (int i = 0; i < 8; ++i) zr[i] = make_uint4(0, 0, 0, 0);
    }
    if (row < M) {
      uint4* yr = reinterpret_cast<uint4*>(y + (long long)row * n + 32 * cj);
      uint4* zr = reinterpret_cast<uint4*>(zo + (long long)row * n + 32 * cj);
#pragma unroll
      for (int i = 0; i < 4; ++i) yr[i] = make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int i = 0; i < 8; ++i) zr[i] = make_uint4(0, 0, 0, 0);
    }
  }
  __device__ void fault(int u, int lane, int s, int (&acc)[2][4][4]) const { fault_word(u, lane, s, acc[0][0][0]); }
  __device__ void fault_word(int u, int lane, int s, int& a) const {
    if ((flags & F_FAULT) && u == 0 && lane == 0 && s == 0) a = (int)((u32)a + (1u << FAULT_SHIFT));
  }
  // the lane's cells of fragment (mi, ni)
  __device__ void out_frag(int ri, int cj, int lane, int mi, int ni, const int (&a)[4]) const {
    const int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int eh = 0; eh < 2; ++eh) {
      const int row = 32 * ri + 16 * mi + g + 8 * eh;
      const float s_a = row < m ? sa[row] : 0.0f;
      const int col = 32 * cj + 8 * ni + 2 * q;
      const int z0 = a[2 * eh], z1 = a[2 * eh + 1];
      if (!(flags & F_NW_TILES)) *reinterpret_cast<int2*>(z + (long long)row * n + col) = make_int2(z0, z1);
      if (row < M) {
        *reinterpret_cast<int2*>(zo + (long long)row * n + col) = make_int2(z0, z1);
        __nv_bfloat162 yv;
        yv.x = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z0), s_a), sw[col]));
        yv.y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z1), s_a), sw[col + 1]));
        *reinterpret_cast<__nv_bfloat162*>(y + (long long)row * n + col) = yv;
        if ((flags & F_DUMP) && row < m && ydump) *reinterpret_cast<__nv_bfloat162*>(ydump + (long long)row * n + col) = yv;
      }
    }
  }
  __device__ void out(int ri, int cj, int lane, const int (&acc)[2][4][4]) const {
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) out_frag(ri, cj, lane, mi, ni, acc[mi][ni]);
  }
  // the split form's half F: fragments ni = F + 2 k
  template <int F>
  __device__ void out_half(int ri, int cj, int lane, const int (&acc)[2][2][4]) const {
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int k = 0; k < 2; ++k) out_frag(ri, cj, lane, mi, F + 2 * k, acc[mi][k]);
  }
};
using OutEpi = OutEpiT<false>;

template <int NW>
__global__ void __launch_bounds__(32 * NW, 2) h1o_units_k(const H1O* __restrict__ F, const H1OStep* __restrict__ S,
                                                          const int8_t* __restrict__ xq, const float* __restrict__ sa, int M,
                                                          const float* __restrict__ sw, __nv_bfloat16* __restrict__ y,
                                                          int* __restrict__ zo, const u64* __restrict__ salt,
                                                          const u64* __restrict__ step_base, u64 nt0, u64 nt1, u64 nt2,
                                                          u64 st0, u64 st1, u64 st2, u64 rt0, u64 rt1, u64 rt2) {
  __shared__ __align__(16) uint8_t smem[NW][2][32 * XS];
  __shared__ __align__(16) uint8_t wbsm[NW][32 * XS];
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
  const OutEpi ep{F->z[p], zo, y, F->ydump, sa, sw, F->n, M, st.m, st.flags, wbsm[wid], salt, step_base,
                  st0, st1, st2, rt0, rt1, rt2};
  h1r_unit<2, OutEpi>(C, v, lane, smem[wid][0], smem[wid][1], sd[wid], nzs[wid][0], F->seeds, nt0, nt1, nt2, F->out[p],
                      (st.flags & F_NW_UNITS) ? 1 : 0, ep);
}

// the block form: a unit a block (`h1b_unit<OutEpi>`), the same values as h1o_units_k's (PL: peeled, z = A Bᵀ from O;
// OCC: 2 for the instantiation `h1o_init` gives the maximum carveout, else 1, the same code)
template <bool PL, int OCC, int BD = 0>
__global__ void __launch_bounds__(BTH, 2) h1o_unitsb_k(const H1O* __restrict__ F, const H1OStep* __restrict__ S,
                                                       const int8_t* __restrict__ xq, const float* __restrict__ sa, int M,
                                                       const float* __restrict__ sw, __nv_bfloat16* __restrict__ y,
                                                       int* __restrict__ zo, const u64* __restrict__ salt,
                                                       const u64* __restrict__ step_base, u64 nt0, u64 nt1, u64 nt2,
                                                       u64 st0, u64 st1, u64 st2, u64 rt0, u64 rt1, u64 rt2,
                                                       const u32* __restrict__ bws, int g) {
  __shared__ H1BShared sh;
  const H1OStep st = *S;
  const int p = st.parity;
  H1Call C = *F->rc[p];
  if ((int)blockIdx.x >= C.units) return;
  C.a = xq;
  C.lda = F->k;
  const OutEpiT<PL, BD> ep{F->z[p], zo, y, F->ydump, sa, sw, F->n, M, st.m, st.flags, nullptr, salt, step_base,
                           st0, st1, st2, rt0, rt1, rt2, bws, g};
  h1b_unit<OutEpiT<PL, BD>, OCC == 2>(C, blockIdx.x, sh, F->seeds, nt0, nt1, nt2, F->out[p],
                                      (st.flags & F_NW_UNITS) ? 1 : 0, ep);
}

// the split form: a unit a block of six warps (`h1s_unit<OutEpi>`), the same values again
template <bool PL, int OCC, int BD = 0>
__global__ void __launch_bounds__(STH, 2) h1o_unitss_k(const H1O* __restrict__ F, const H1OStep* __restrict__ S,
                                                       const int8_t* __restrict__ xq, const float* __restrict__ sa, int M,
                                                       const float* __restrict__ sw, __nv_bfloat16* __restrict__ y,
                                                       int* __restrict__ zo, const u64* __restrict__ salt,
                                                       const u64* __restrict__ step_base, u64 nt0, u64 nt1, u64 nt2,
                                                       u64 st0, u64 st1, u64 st2, u64 rt0, u64 rt1, u64 rt2,
                                                       const u32* __restrict__ bws, int g) {
  __shared__ H1SShared sh;
  const H1OStep st = *S;
  const int p = st.parity;
  H1Call C = *F->rc[p];
  if ((int)blockIdx.x >= C.units) return;
  C.a = xq;
  C.lda = F->k;
  const OutEpiT<PL, BD> ep{F->z[p], zo, y, F->ydump, sa, sw, F->n, M, st.m, st.flags, nullptr, salt, step_base,
                           st0, st1, st2, rt0, rt1, rt2, bws, g};
  h1s_unit<OutEpiT<PL, BD>>(C, blockIdx.x, sh, F->seeds, nt0, nt1, nt2, F->out[p], (st.flags & F_NW_UNITS) ? 1 : 0, ep);
}

// J10 banded (measured only; labelled noise=j10-banded, j10b_ref.py's reading of SharedNoise.lean): band (ri, b)
// of g column tiles reads one set of row streams, its leader unit ri ncb + b g's h1tile rows (seed_j's blocks l RB + c),
// written here once a step into bws before the call's units, which hash only their own columns.  One warp a (live row
// block ri, band b, J consecutive chunks), lane l row l; every lane derives the leader's seed_j (`unit_seed`, the
// position base + step + ri ncb + b g), so the pass reads root(A) as the units do, after the call's A table.
__global__ void __launch_bounds__(128) h1o_bandrows_k(const H1O* __restrict__ F, const H1OStep* __restrict__ S,
                                                      const u64* __restrict__ salt, const u64* __restrict__ step_base,
                                                      u32* __restrict__ bws, int g, int J, u64 nt0, u64 nt1, u64 nt2,
                                                      u64 st0, u64 st1, u64 st2) {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  const H1Call C = *F->rc[S->parity];
  const int lane = threadIdx.x & 31;
  const long long item = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nrb = C.units / C.ncb, nl = (C.ncb + g - 1) / g, RB = (C.k >> 9) + 1, per = (RB + J - 1) / J;
  if (item >= (long long)nrb * nl * per) return;
  const int L = (int)(item / per), c0 = (int)(item % per) * J, ri = L / nl, b = L % nl;
  if (!C.live[ri]) return;
  u64 sd[8], mid[8];
  unit_seed(C, C.base + *step_base + (u64)(ri * C.ncb + b * g), salt, st0, st1, st2, sd);
  noise_mid(sd, nt0, nt1, nt2, mid);
  for (int c = c0; c < c0 + J && c < RB; ++c) {
    u32 h[16];
    nblock_mid(mid, sd, (u64)lane * RB + c, nt0, nt1, nt2, h);
    uint4* o = reinterpret_cast<uint4*>(bws + 16ull * ((unsigned long long)L * 32 * RB + (unsigned long long)lane * RB + c));
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
  }
}

// the reference: unit u = ri * ncb + cj of call C (A rows from C.a, rows >= the slot's m already zero), z_ref [T, N]
// wbc (or null): [wb⁺ count, wb⁻ count, xa count] added over the call's live units
__global__ void __launch_bounds__(32) h1o_ref_k(const H1Call* __restrict__ call, const u64* __restrict__ seeds, u64 nt0,
                                                u64 nt1, u64 nt2, int* __restrict__ zref, int n,
                                                unsigned long long* __restrict__ wbc) {
  __shared__ __align__(16) uint8_t Xs[32][516];
  __shared__ __align__(16) uint8_t Ws[32][516];
  __shared__ int8_t Us[32][32];
  __shared__ int Cs[32][33];  // [row][column]: sum over wrapped entries of ±X, column j's written by lane j alone
  const H1Call C = *call;
  const int u = blockIdx.x, lane = threadIdx.x;
  if (u >= C.units) return;
  const int ri = u / C.ncb, cj = u % C.ncb;
  int* zr = zref + (long long)(32 * ri + lane) * n + 32 * cj;
  if (!C.live[ri]) {
    for (int j = 0; j < 32; ++j) zr[j] = 0;
    return;
  }
  for (int j = 0; j < 32; ++j) Cs[lane][j] = 0;
  unsigned long long np = 0, nn = 0, nx = 0;
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
      const int bit = (int)((hr[s16] >> t) & 1u);
      nx += (a == 127) & bit;
      Xs[lane][kk] = (uint8_t)(a + bit);
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
    for (int kk = 0; kk < 512; ++kk) {
      const int s16 = kk / 32, t = kk % 32;
      const int sv = (int)brow[512 * ch + kk] + (int)((hc[s16] >> t) & 1u) + (int)Us[lane][t];
      const int w = sv > 127 ? 1 : sv < -128 ? -1 : 0;
      if (w) {
        if (w > 0) ++np; else ++nn;
        for (int i = 0; i < 32; ++i) Cs[i][lane] += w * (int)(int8_t)Xs[i][kk];
      }
    }
  }
  __syncwarp();
  for (int j = 0; j < 32; ++j) {
    u32 o = (u32)acc[j];
    for (int r = 0; r < 32; ++r) o -= (u32)((int)Us[j][r] * Y[r]);
    zr[j] = (int)(o + ((u32)Cs[lane][j] << 8));
  }
  if (wbc) {
    atomicAdd(wbc, np);
    atomicAdd(wbc + 1, nn);
    atomicAdd(wbc + 2, nx);
  }
}

constexpr int NWO = 4;

// one served call: xq [M, K] s8 and sa [M] f32 (row_quant's), sw [N] f32 (the layer's), y [M, N] bf16 and zo [M, N] s32
// (the call's outputs); F (the call's entry), S, the window's salt and the step's base position on the device.  The
// weights are the call table's.  The caller's h1c_a_call (the slot and its A table) runs first.
extern "C" int h1o_units_launch(const H1O* F, const H1OStep* S, int units, int n, int k, const int8_t* xq, const float* sa,
                                int M, const float* sw, void* y, int* zo, const u64* salt, const u64* step_base,
                                cudaStream_t stream) {
  if (M <= 0 || n % 32 || k % 512 || k < 512 || units <= 0 || !salt || !step_base) return 1;
  u64 nt[3], st[3], rt[3];
  tag3("verity/pouw/noise/v1", nt);
  tag3("verity/pouw/seed/v2", st);
  tag3("verity/pouw/receipt/v1", rt);
  h1o_units_k<NWO><<<(unsigned)((units + NWO - 1) / NWO), 32 * NWO, 0, stream>>>(
      F, S, xq, sa, M, sw, (__nv_bfloat16*)y, zo, salt, step_base, nt[0], nt[1], nt[2], st[0], st[1], st[2], rt[0], rt[1],
      rt[2]);
  return (int)cudaGetLastError();
}

template <bool PL, int OCC>
static int unitsb(const H1O* F, const H1OStep* S, int units, int n, int k, const int8_t* xq, const float* sa, int M,
                  const float* sw, void* y, int* zo, const u64* salt, const u64* step_base, cudaStream_t stream) {
  if (M <= 0 || n % 32 || k % 512 || k < 512 || units <= 0 || !salt || !step_base) return 1;
  u64 nt[3], st[3], rt[3];
  tag3("verity/pouw/noise/v1", nt);
  tag3("verity/pouw/seed/v2", st);
  tag3("verity/pouw/receipt/v1", rt);
  h1o_unitsb_k<PL, OCC><<<(unsigned)units, BTH, 0, stream>>>(F, S, xq, sa, M, sw, (__nv_bfloat16*)y, zo, salt, step_base,
                                                             nt[0], nt[1], nt[2], st[0], st[1], st[2], rt[0], rt[1], rt[2],
                                                             nullptr, 1);
  return (int)cudaGetLastError();
}

template <bool PL, int OCC>
static int unitss(const H1O* F, const H1OStep* S, int units, int n, int k, const int8_t* xq, const float* sa, int M,
                  const float* sw, void* y, int* zo, const u64* salt, const u64* step_base, cudaStream_t stream) {
  if (M <= 0 || n % 32 || k % 512 || k < 512 || k > 16384 || units <= 0 || !salt || !step_base) return 1;
  u64 nt[3], st[3], rt[3];
  tag3("verity/pouw/noise/v1", nt);
  tag3("verity/pouw/seed/v2", st);
  tag3("verity/pouw/receipt/v1", rt);
  h1o_unitss_k<PL, OCC><<<(unsigned)units, STH, 0, stream>>>(F, S, xq, sa, M, sw, (__nv_bfloat16*)y, zo, salt, step_base,
                                                             nt[0], nt[1], nt[2], st[0], st[1], st[2], rt[0], rt[1], rt[2],
                                                             nullptr, 1);
  return (int)cudaGetLastError();
}

#define H1O_UNITS_EXPORT(NAME, FORM, PL, OCC)                                                                              \
  extern "C" int NAME(const H1O* F, const H1OStep* S, int units, int n, int k, const int8_t* xq, const float* sa, int M, \
                      const float* sw, void* y, int* zo, const u64* salt, const u64* step_base, cudaStream_t stream) {   \
    return FORM<PL, OCC>(F, S, units, n, k, xq, sa, M, sw, y, zo, salt, step_base, stream);                               \
  }
// h1o_units_launch's call in the block form, and peeled (z = A Bᵀ from O); in the split form (K <= 16384), and peeled;
// and each again with the maximum carveout (after `h1o_init`)
H1O_UNITS_EXPORT(h1o_unitsb_launch, unitsb, false, 1)
H1O_UNITS_EXPORT(h1o_unitsbp_launch, unitsb, true, 1)
H1O_UNITS_EXPORT(h1o_unitss_launch, unitss, false, 1)
H1O_UNITS_EXPORT(h1o_unitssp_launch, unitss, true, 1)
H1O_UNITS_EXPORT(h1o_unitsb2_launch, unitsb, false, 2)
H1O_UNITS_EXPORT(h1o_unitsb2p_launch, unitsb, true, 2)
H1O_UNITS_EXPORT(h1o_unitss2_launch, unitss, false, 2)
H1O_UNITS_EXPORT(h1o_unitss2p_launch, unitss, true, 2)

// J10 banded, peeled (measured only; noise=j10-banded, j10b_ref.py's reading of SharedNoise.lean): bws holds
// h1o_band_bytes(units, n, k, g) bytes, the call's own; form 0 block2 (OCC 2) and 3 block with the rows read in the
// segment loop (NoEpi::BAND 1), 1 split and 2 split2 (staged), 4 block2 and 5 block staged (BAND 2); mode 0 the band-rows
// pass then the units, 1 the units alone (rows from the last pass), 2 the pass alone, 3 mode 0 with the units launched as
// the pass's programmatic dependents (their seeds, midstates and column tiles overlap the pass; band_wait before the rows)
static int band_J(int units, int ncb, int k, int g) {
  const long long nrb = units / ncb, nl = (ncb + g - 1) / g, RB = (k >> 9) + 1;
  int J = 1;
  while (J < RB && nrb * nl * ((RB + J - 1) / J) > 16384) ++J;
  return J;
}

extern "C" unsigned long long h1o_band_bytes(int units, int n, int k, int g) {
  const int ncb = n / 32;
  return 64ull * (unsigned long long)(units / ncb) * ((ncb + g - 1) / g) * 32 * ((k >> 9) + 1);
}

using BandUnitsK = void (*)(const H1O*, const H1OStep*, const int8_t*, const float*, int, const float*, __nv_bfloat16*,
                            int*, const u64*, const u64*, u64, u64, u64, u64, u64, u64, u64, u64, u64, const u32*, int);
static BandUnitsK band_kernel(int form) {
  switch (form) {
    case 0: return h1o_unitsb_k<true, 2, 1>;
    case 1: return h1o_unitss_k<true, 1, 2>;
    case 2: return h1o_unitss_k<true, 2, 2>;
    case 3: return h1o_unitsb_k<true, 1, 1>;
    case 4: return h1o_unitsb_k<true, 2, 2>;
    case 5: return h1o_unitsb_k<true, 1, 2>;
    default: return nullptr;
  }
}
static bool band_split(int form) { return form == 1 || form == 2; }

extern "C" int h1o_band_launch(int form, int mode, const H1O* F, const H1OStep* S, int units, int n, int k,
                               const int8_t* xq, const float* sa, int M, const float* sw, void* y, int* zo, const u64* salt,
                               const u64* step_base, u32* bws, int g, cudaStream_t stream) {
  const BandUnitsK kf = band_kernel(form);
  if (M <= 0 || n % 32 || k % 512 || k < 512 || units <= 0 || !salt || !step_base || !bws || g < 1 || !kf || mode < 0
      || mode > 3 || band_split(form) && k > 16384)
    return 1;
  u64 nt[3], st[3], rt[3];
  tag3("verity/pouw/noise/v1", nt);
  tag3("verity/pouw/seed/v2", st);
  tag3("verity/pouw/receipt/v1", rt);
  if (mode != 1) {
    const int ncb = n / 32, J = band_J(units, ncb, k, g), RB = (k >> 9) + 1;
    const long long items = (long long)(units / ncb) * ((ncb + g - 1) / g) * ((RB + J - 1) / J);
    h1o_bandrows_k<<<(unsigned)((items * 32 + 127) / 128), 128, 0, stream>>>(F, S, salt, step_base, bws, g, J, nt[0], nt[1],
                                                                            nt[2], st[0], st[1], st[2]);
  }
  if (mode == 2) return (int)cudaGetLastError();
  if (mode == 3) {
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)units);
    cfg.blockDim = dim3(band_split(form) ? STH : BTH);
    cfg.stream = stream;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, kf, F, S, xq, sa, M, sw, (__nv_bfloat16*)y, zo, salt, step_base, nt[0],
                                             nt[1], nt[2], st[0], st[1], st[2], rt[0], rt[1], rt[2], (const u32*)bws, g);
    return e != cudaSuccess ? (int)e : (int)cudaGetLastError();
  }
  kf<<<(unsigned)units, band_split(form) ? STH : BTH, 0, stream>>>(F, S, xq, sa, M, sw, (__nv_bfloat16*)y, zo, salt,
                                                                    step_base, nt[0], nt[1], nt[2], st[0], st[1], st[2],
                                                                    rt[0], rt[1], rt[2], bws, g);
  return (int)cudaGetLastError();
}

// h1o_init for the banded kernels (the OCC = 2 ones' carveout), and their blocks an SM by h1o_band_launch's form
extern "C" int h1o_band_init() {
  for (int f : {0, 2, 4}) {
    const cudaError_t e = cudaFuncSetAttribute((const void*)band_kernel(f), cudaFuncAttributePreferredSharedMemoryCarveout,
                                               (int)cudaSharedmemCarveoutMaxShared);
    if (e != cudaSuccess) return (int)e;
  }
  return 0;
}

extern "C" int h1o_band_occupancy(int form) {
  const BandUnitsK kf = band_kernel(form);
  int nb = 0;
  if (!kf || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)kf, band_split(form) ? STH : BTH, 0) != cudaSuccess)
    return -1;
  return nb;
}

// once a process, before any capture: the OCC = 2 kernels' preferred carveout to the maximum shared memory
extern "C" int h1o_init() {
  const void* fs[] = {(const void*)h1o_unitsb_k<false, 2>, (const void*)h1o_unitsb_k<true, 2>,
                      (const void*)h1o_unitss_k<false, 2>, (const void*)h1o_unitss_k<true, 2>};
  for (const void* f : fs) {
    const cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout,
                                               (int)cudaSharedmemCarveoutMaxShared);
    if (e != cudaSuccess) return (int)e;
  }
  return 0;
}

// blocks an SM the occupancy calculator gives a units form: which 0..3 = b, bp, s, sp; + 4 for the OCC = 2 kernels
extern "C" int h1o_occupancy(int which) {
  const void* fs[] = {(const void*)h1o_unitsb_k<false, 1>, (const void*)h1o_unitsb_k<true, 1>,
                      (const void*)h1o_unitss_k<false, 1>, (const void*)h1o_unitss_k<true, 1>,
                      (const void*)h1o_unitsb_k<false, 2>, (const void*)h1o_unitsb_k<true, 2>,
                      (const void*)h1o_unitss_k<false, 2>, (const void*)h1o_unitss_k<true, 2>};
  if (which < 0 || which > 7) return -1;
  int nb = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fs[which], (which & 2) ? STH : BTH, 0) != cudaSuccess) return -1;
  return nb;
}

extern "C" int h1o_ref_launch(const H1Call* call_dev, int units, const u64* seeds, int* zref, int n,
                              unsigned long long* wbc, cudaStream_t stream) {
  u64 nt[3];
  tag3("verity/pouw/noise/v1", nt);
  h1o_ref_k<<<(unsigned)units, 32, 0, stream>>>(call_dev, seeds, nt[0], nt[1], nt[2], zref, n, wbc);
  return (int)cudaGetLastError();
}

// The per-call challenger round's measurement (labelled `percall-rtt`, measured only): between a call's A table and its units, the A
// root to the host, a host function that holds the stream one challenger round, and a 64-byte coin back to the device,
// all three on the caller's stream, so a captured graph keeps them (memcpy nodes and a host node).  The round is
// simulated: the host function spins for the next of the waits h1o_percall_init took (the measured round trips, replayed
// in order), and the coin it writes is a stand-in (the root, xored) that no unit reads; the stream order holds the units
// behind the copy, as reading it would.
struct PcArg {
  const unsigned char* root;
  unsigned char* coin;
};
static unsigned char* pc_host = nullptr;   // pinned, per linear: the root's 64 bytes, then the coin's
static PcArg* pc_arg = nullptr;
static long long* pc_ns = nullptr;
static int pc_l = 0, pc_n = 0;
static unsigned long long pc_i = 0, pc_done = 0;

static void CUDART_CB pc_wait(void* p) {
  const PcArg* a = (const PcArg*)p;
  const unsigned long long i = __atomic_fetch_add(&pc_i, 1ull, __ATOMIC_RELAXED);
  const long long ns = pc_ns[i % (unsigned long long)pc_n];
  if (ns > 0) {
    timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    do clock_gettime(CLOCK_MONOTONIC, &t);
    while ((t.tv_sec - t0.tv_sec) * 1000000000ll + (t.tv_nsec - t0.tv_nsec) < ns);
  }
  for (int j = 0; j < 64; ++j) a->coin[j] = a->root[j] ^ 0x5a;
  __atomic_fetch_add(&pc_done, 1ull, __ATOMIC_RELEASE);
}

// before any capture: L linears' pinned slots, and the n waits (ns) the host function replays
extern "C" int h1o_percall_init(int L, const long long* ns, int n) {
  if (L <= 0 || n <= 0 || pc_host) return 1;
  if (cudaHostAlloc((void**)&pc_host, (size_t)L * 128, cudaHostAllocDefault) != cudaSuccess) return 2;
  pc_arg = new PcArg[L];
  for (int w = 0; w < L; ++w) pc_arg[w] = {pc_host + 128 * w, pc_host + 128 * w + 64};
  pc_ns = new long long[n];
  for (int i = 0; i < n; ++i) pc_ns[i] = ns[i];
  pc_l = L, pc_n = n;
  return 0;
}

extern "C" int h1o_percall_launch(int w, const void* root_dev, void* coin_dev, cudaStream_t stream) {
  if (!pc_host || w < 0 || w >= pc_l) return 1;
  cudaMemcpyAsync(pc_host + 128 * w, root_dev, 64, cudaMemcpyDeviceToHost, stream);
  cudaLaunchHostFunc(stream, pc_wait, &pc_arg[w]);
  cudaMemcpyAsync(coin_dev, pc_host + 128 * w + 64, 64, cudaMemcpyHostToDevice, stream);
  return (int)cudaGetLastError();
}

// the rounds the host function has served so far (graph replays included)
extern "C" unsigned long long h1o_percall_done() { return __atomic_load_n(&pc_done, __ATOMIC_ACQUIRE); }

extern "C" int h1o_struct_bytes(int which) { return which == 0 ? (int)sizeof(H1O) : (int)sizeof(H1OStep); }
extern "C" int h1o_fault_shift() { return FAULT_SHIFT; }
