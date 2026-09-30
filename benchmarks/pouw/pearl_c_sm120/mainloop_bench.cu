// Bench for mainloop_sm120.cuh: GPU 1's kernels (the gated references) and the mainloop's CT kernels in one cubin.
#include "pearl_c_sm120.cu"
#include "mainloop_sm120.cuh"

// random E4M3 codes without NaN: p0 out (u32 words), i0 words, i1 seed
extern "C" __global__ void ml_gen_e4m3(const Args a) {
  const size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= (size_t)a.i[0]) return;
  uint32_t h = mix32((uint32_t)a.i[1] ^ mix32((uint32_t)w * 2u + 1u));
  uint32_t o = 0;
  for (int b = 0; b < 4; b++) {
    uint32_t c = (h >> (8 * b)) & 0xFF;
    if ((c & 0x7F) == 0x7F) c ^= 1;
    o |= c << (8 * b);
  }
  reinterpret_cast<uint32_t*>(a.p[0])[w] = o;
}

// The tail's bytes against their source: the bench's P_A and P_B maps are A's and B's, so a tile's tail holds A's TM rows and
// B's BN rows at depth [0, 128), in the SWIZZLE_128B layout; true if any byte differs
template <int TM, int BN, int NCT>
DEV bool ml_tail_bad(uint32_t p, const uint8_t* A, const uint8_t* B, int K, int tm, int tn) {
  bool bad = false;
  for (int i = threadIdx.x; i < (TM + BN) * 8; i += NCT) {
    const int r = i / 8, c = i % 8, side = r >= TM, row = side ? r - TM : r;
    const uint8_t* g = side ? B + (size_t)(tn * BN + row) * K : A + (size_t)(tm * TM + row) * K;
    const uint4 want = *reinterpret_cast<const uint4*>(g + 16 * c);
    uint4 got;
    const uint32_t at = p + (side ? TM * 128 : 0) + row * 128 + ((c ^ (row & 7)) << 4);
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(got.x), "=r"(got.y), "=r"(got.z), "=r"(got.w) : "r"(at));
    bad |= got.x != want.x || got.y != want.y || got.z != want.z || got.w != want.w;
  }
  return bad;
}

// C~ by the mainloop: p0 A, p1 B, p2 C (FP32), p10 the Maps; i0 M, i1 N, i2 K, i3 raster group
// U: stages per K-loop iteration, or 0 for the skewed schedule.  TL: the tail held through the epilogue and checked
// (ml_tail_bad); a thread that finds a wrong byte writes a NaN word into C~, which every gate refuses
template <int TM, int BN, int G, int STAGES, int U, bool LATE, int CTAS = 1, int RP = 40, int WM = 64, bool TL = false>
DEV void ml_ct_body(const Args& a) {
  using L = ml::Mainloop<TM, BN, G, STAGES, TL, CTAS, RP, WM>;
  extern __shared__ __align__(1024) uint8_t ml_smem[];
  L ml(ml_smem);
  const int M = a.i[0], N = a.i[1], K = a.i[2], group = a.i[3], warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.roles(reinterpret_cast<const ml::Maps*>(a.p[10]), M, N, K, group);
  float* C = reinterpret_cast<float*>(a.p[2]);
  const int tiles_m = M / TM, tiles_n = N / BN, wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3;
  if (a.i[4] && 2 * wm >= L::WARPS_M) {   // i4: cycles the second half of the warp rows start late (their SMSP partners run ahead)
    const long long t0 = clock64();
    while (clock64() - t0 < a.i[4]) {}
  }
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    ml::raster(t, tiles_m, tiles_n, group, tm, tn);
    float acc[L::MI][L::NI][4], total[L::MI][L::NI][4];
    if constexpr (U == 0) ml.tile_skew(K, acc, total);
    else ml.template tile<U, LATE>(K, acc, total);
    if constexpr (TL) {   // i5: 1 checks the tail against depth [128, 256) instead (the negative control), 2 skips the check
      const uint32_t p = ml.tail_wait();
      const int off = a.i[5] == 1 ? 128 : 0;
      if (a.i[5] != 2 && ml_tail_bad<TM, BN, L::NCT>(p, reinterpret_cast<const uint8_t*>(a.p[0]) + off,
                                                     reinterpret_cast<const uint8_t*>(a.p[1]) + off, K, tm, tn))
        total[0][0][0] = __uint_as_float(0x7fc0deadu);
    }
    const int row0 = tm * TM + wm * L::WM + g, col0 = tn * BN + wn * L::WN + 2 * q;
#pragma unroll
    for (int mi = 0; mi < L::MI; mi++)
#pragma unroll
      for (int ni = 0; ni < L::NI; ni++) {
        const int row = row0 + 16 * mi, col = col0 + 8 * ni;
        store2(C + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1]);
        store2(C + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3]);
      }
    if constexpr (TL) ml.tail_release();
  }
}
#define ML_CT_W(NAME, TM, BN, G, ST, U, LATE, CTAS, RP, WM)                                                              \
  extern "C" __global__ void __launch_bounds__((ml::Mainloop<TM, BN, G, ST, false, CTAS, RP, WM>::NT), CTAS) NAME(const Args a) { \
    ml_ct_body<TM, BN, G, ST, U, LATE, CTAS, RP, WM>(a);                                                                      \
  }                                                                                                                  \
  extern "C" __global__ void NAME##_smem(int* out) { *out = ml::Mainloop<TM, BN, G, ST, false, CTAS, RP, WM>::SMEM; }
#define ML_CT_R(NAME, TM, BN, G, ST, U, LATE, CTAS, RP) ML_CT_W(NAME, TM, BN, G, ST, U, LATE, CTAS, RP, 64)
#define ML_CT_T(NAME, TM, BN, G, ST, U, LATE, CTAS) ML_CT_R(NAME, TM, BN, G, ST, U, LATE, CTAS, 40)
#define ML_CT(NAME, G, ST, U, LATE) ML_CT_T(NAME, 128, 128, G, ST, U, LATE, 1)
ML_CT(ml_ct, 4, 3, 1, true)
ML_CT(ml_ct_u, 0, 3, 1, true)
ML_CT(ml_ct_e, 4, 3, 1, false)
ML_CT(ml_ct_u_e, 0, 3, 1, false)
ML_CT(ml_ct_x2, 4, 3, 2, true)
ML_CT(ml_ct_x2e, 4, 3, 2, false)
ML_CT(ml_ct_u_x2, 0, 3, 2, true)
// the producer at 24 registers, 240 to each consumer
ML_CT_R(ml_ct_x2r, 128, 128, 4, 3, 2, true, 1, 24)
ML_CT_R(ml_ct_x2er, 128, 128, 4, 3, 2, false, 1, 24)
ML_CT_R(ml_ct_r, 128, 128, 4, 3, 1, true, 1, 24)
ML_CT(ml_cts, 4, 3, 0, true)
ML_CT(ml_cts_u, 0, 3, 0, true)
// the other tile shapes GPU 1's kernel instantiates (K3 as two 4-warp blocks per SM; decode's 64 x 64; the 32-column words)
ML_CT_T(ml_ct_128x64_2, 128, 64, 4, 2, 1, true, 2)
ML_CT_T(ml_ct_128x64, 128, 64, 4, 3, 1, true, 1)
ML_CT_T(ml_ct_64x64, 64, 64, 4, 3, 1, true, 1)
ML_CT_T(ml_ct_128x32, 128, 32, 4, 3, 1, true, 1)
// K3's two 4-warp blocks per SM with the producer at 24 registers (232 to each consumer); 128 x 128 at 2 stages (a stage's
// 32 KB free for the epilogue)
ML_CT_R(ml_ct_r_128x64_2, 128, 64, 4, 2, 1, true, 2, 24)
ML_CT_R(ml_ct_r_u_128x64_2, 128, 64, 0, 2, 1, true, 2, 24)
ML_CT_R(ml_ct_s2, 128, 128, 4, 2, 1, true, 1, 40)
ML_CT_R(ml_ct_s2_x2r, 128, 128, 4, 2, 2, true, 1, 24)
// decode's 64 x 64 and the words' 128 x 32 as 8 warps of 16 x 32 (GPU 1's warp tiles there)
ML_CT_W(ml_ct_w16_64x64, 64, 64, 4, 3, 1, true, 1, 40, 16)
ML_CT_W(ml_ct_w16_u_64x64, 64, 64, 0, 3, 1, true, 1, 40, 16)
ML_CT_W(ml_ct_w16_128x32, 128, 32, 4, 3, 1, true, 1, 40, 16)
ML_CT_W(ml_ct_w16_x2_64x64, 64, 64, 4, 3, 2, true, 1, 40, 16)
ML_CT_W(ml_ct_x2r_128x64_2, 128, 64, 4, 2, 2, true, 2, 24, 64)
// the recommended v1 and v2 with the tail (P_A, P_B) held through the epilogue, as GPU 1's peel needs it
#define ML_CT_TL(NAME, G, U, RP)                                                                                          \
  extern "C" __global__ void __launch_bounds__((ml::Mainloop<128, 128, G, 3, true, 1, RP>::NT), 1) NAME(const Args a) { \
    ml_ct_body<128, 128, G, 3, U, true, 1, RP, 64, true>(a);                                                             \
  }                                                                                                                     \
  extern "C" __global__ void NAME##_smem(int* out) { *out = ml::Mainloop<128, 128, G, 3, true, 1, RP>::SMEM; }
ML_CT_TL(ml_ct_x2r_tail, 4, 2, 24)
ML_CT_TL(ml_ct_u_x2_tail, 0, 2, 40)
