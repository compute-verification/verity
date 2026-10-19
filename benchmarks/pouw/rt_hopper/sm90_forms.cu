// RT-HOPPER-1 on the H200 (sm_90a), the census's s8-rate forms besides the honest one: IGMMA_S8_N8 to N192,
// IMMA16832_S8 and IMMA16816_S8 (mma.sync m16n8k32, m16n8k16) and BMMA168256_B1 (mma.sync m16n8k256 b1). The census
// prices each on DC-90's joint rows at the lane default, as it does RS, and offering any one but IMMA16816 beside the
// honest form takes DC-90-ri from 0.615% to 3-29% (`forms_gap.py --menu`). Each runs sm90_rs.cu's four loops at one
// N256 group's MACs a step (N192 at three quarters of them; 128 mma.sync a warp, 256 at k16): alone, beside the LI
// fold, and at the IMAD and LOP3 points, with the SS N256 references, so its joint coefficients can be fitted as RS's
// were. The FP wgmma kinds at N256 (f16, bf16, tf32, e4m3) run the same four loops (`wf`). rs.sh builds it with
// RS_SRC=sm90_forms.cu; rs.py reads it.
#define RS_NO_MAIN
#include "sm90_rs.cu"

// NACC accumulator registers a thread: N = 2 * NACC
template <int NACC, int G>
DC_DEV void group_n(int (&d)[NACC], const Tiles& t) {
  wgmma_fence();
#pragma unroll
  for (int g = 0; g < G; g++) {
    if constexpr (NACC == 96) wgmma_s8_n192(d, t.da, t.db, g ? 1 : 0);
    if constexpr (NACC == 64) wgmma_s8_n128(d, t.da, t.db, g ? 1 : 0);
    if constexpr (NACC == 32) wgmma_s8_n64(d, t.da, t.db, g ? 1 : 0);
    if constexpr (NACC == 16) wgmma_s8_n32(d, t.da, t.db, g ? 1 : 0);
    if constexpr (NACC == 8) wgmma_s8_n16(d, t.da, t.db, g ? 1 : 0);
    if constexpr (NACC == 4) wgmma_s8_n8(d, t.da, t.db, g ? 1 : 0);
  }
  wgmma_commit();
}

// L: 0 alone (two groups in flight), 1 the LI fold, 2 IMAD r256, 3 LOP3 r256
template <int NACC, int G, int L>
DC_DEV void wn(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles(sm, a.mode);
  int d[NACC] = {};
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
    group_n<NACC, G>(d, t);
    if constexpr (L == 0) wgmma_wait<1>();
    if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 256; c++) {
        l.op<L_LOP3>(c);
        l.op<L_IMAD>(c);
      }
#pragma unroll
      for (int s = 0; s < 4; s++) {
        l.op<L_LOP3>(0);
        l.op<L_LOP3>(2);
        l.op<L_LOP3>(4);
        l.op<L_LOP3>(6);
        l.op<L_REDUX>(0);
      }
    }
    if constexpr (L == 2) lane_ops<L_IMAD, 256>(l);
    if constexpr (L == 3) lane_ops<L_LOP3, 256>(l);
    if constexpr (L != 0) wgmma_wait<0>();
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}

// mma.sync, operands loop-invariant, on 8 accumulator tiles, one N256 group's warp share a step: S 0 m16n8k32 s8 (128
// a step), 1 m16n8k16 s8 (256), 2 m16n8k256 b1 and.popc (128, at BGMMA's convention: a 32nd of its warp share)
template <int S>
DC_DEV void mma_sync(uint32_t (&c)[4], const uint32_t (&x)[4], const uint32_t (&y)[2]) {
  if constexpr (S == 0)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(y[0]), "r"(y[1]));
  if constexpr (S == 1)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(x[0]), "r"(x[1]), "r"(y[0]));
  if constexpr (S == 2)
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(y[0]), "r"(y[1]));
}

template <int L, int S = 0>
DC_DEV void im(const OpArgs& a) {
  uint32_t c[8][4] = {}, x[4], y[2];
#pragma unroll
  for (int i = 0; i < 4; i++) x[i] = dc_mix32(threadIdx.x * 4u + i + (uint32_t)a.mode);
  y[0] = dc_mix32(threadIdx.x + 0x51u);
  y[1] = dc_mix32(threadIdx.x + 0x93u);
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int i = 0; i < (S == 1 ? 256 : 128); i++) mma_sync<S>(c[i & 7], x, y);
    if constexpr (L == 1) {
#pragma unroll
      for (int q = 0; q < 256; q++) {
        l.op<L_LOP3>(q);
        l.op<L_IMAD>(q);
      }
#pragma unroll
      for (int s = 0; s < 4; s++) {
        l.op<L_LOP3>(0);
        l.op<L_LOP3>(2);
        l.op<L_LOP3>(4);
        l.op<L_LOP3>(6);
        l.op<L_REDUX>(0);
      }
    }
    if constexpr (L == 2) lane_ops<L_IMAD, 256>(l);
    if constexpr (L == 3) lane_ops<L_LOP3, 256>(l);
  }
  uint32_t s = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) s ^= c[i][0] ^ c[i][1] ^ c[i][2] ^ c[i][3];
  OP_END
  OP_SINK(s ^ l.fold() ^ threadIdx.x)
}

// The census's FP wgmma kinds at N256, each into FP32 on the same 32-byte k-major tiles (a row's 32 bytes are f16 and
// bf16's k16, tf32's k8 and e4m3's k32): T 0 f16, 1 bf16, 2 tf32, 3 e4m3, G of them a step for one N256 s8 group's MACs
// (8, 8, 16, 4). Each is exact for s8 products at split 1 (f16, bf16, tf32) or 3 (e4m3), forms_gap.py's FP.
template <int T>
DC_DEV void mma_fp(float (&d)[128], const Tiles& t, int sd) {
  if constexpr (T == 0) wgmma_f16_n256(d, t.da, t.db, sd);
  if constexpr (T == 1) wgmma_bf16_n256(d, t.da, t.db, sd);
  if constexpr (T == 2) wgmma_tf32_n256(d, t.da, t.db, sd);
  if constexpr (T == 3) wgmma_e4m3_n256(d, t.da, t.db, sd);
}

// fill_tiles with ../dc/probes/sm90_wgmma.cu's operand words for each FP type, as the census's lines were timed: finite,
// f16 and bf16 in ±[0.5, 1), tf32 in [1, 2), e4m3 with no NaN code
template <int T>
DC_DEV Tiles fill_fp(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x) {
    const uint32_t x = dc_mix32(i * 0x9E3779B9u + 0x1234567u);
    w[i] = mode == 1 ? 0u
           : T == 0  ? (x & 0x83FF83FFu) | 0x38003800u
           : T == 1  ? (x & 0x807F807Fu) | 0x3F003F00u
           : T == 2  ? (x & 0x007FFFFFu) | 0x3F800000u
                     : x & 0x77777777u;
  }
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB))};
}

// wn's four loops on an FP form
template <int T, int G, int L>
DC_DEV void wf(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_fp<T>(sm, a.mode);
  float d[128] = {};
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
    wgmma_fence();
#pragma unroll
    for (int g = 0; g < G; g++) mma_fp<T>(d, t, g ? 1 : 0);
    wgmma_commit();
    if constexpr (L == 0) wgmma_wait<1>();
    if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 256; c++) {
        l.op<L_LOP3>(c);
        l.op<L_IMAD>(c);
      }
#pragma unroll
      for (int s = 0; s < 4; s++) {
        l.op<L_LOP3>(0);
        l.op<L_LOP3>(2);
        l.op<L_LOP3>(4);
        l.op<L_LOP3>(6);
        l.op<L_REDUX>(0);
      }
    }
    if constexpr (L == 2) lane_ops<L_IMAD, 256>(l);
    if constexpr (L == 3) lane_ops<L_LOP3, 256>(l);
    if constexpr (L != 0) wgmma_wait<0>();
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  OP_END
  uint32_t s = 0;
#pragma unroll
  for (int i = 0; i < 128; i++) s ^= __float_as_uint(d[i]);
  OP_SINK(s ^ l.fold() ^ threadIdx.x)
}

OP_KERNEL(wg_s8_n128_a1, 256) { wn<64, 8, 0>(a); }
OP_KERNEL(wgli_redux_n128, 256) { wn<64, 8, 1>(a); }
OP_KERNEL(wgmix_imad_r256_n128, 256) { wn<64, 8, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_n128, 256) { wn<64, 8, 3>(a); }
OP_KERNEL(wg_s8_n64_a1, 256) { wn<32, 16, 0>(a); }
OP_KERNEL(wgli_redux_n64, 256) { wn<32, 16, 1>(a); }
OP_KERNEL(wgmix_imad_r256_n64, 256) { wn<32, 16, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_n64, 256) { wn<32, 16, 3>(a); }
OP_KERNEL(imma_s8_a1, 256) { im<0>(a); }
OP_KERNEL(wgli_redux_im, 256) { im<1>(a); }
OP_KERNEL(wgmix_imad_r256_im, 256) { im<2>(a); }
OP_KERNEL(wgmix_lop3_r256_im, 256) { im<3>(a); }
// the rest of the census's s8-rate forms; N192 runs 4 a step, three quarters of a N256 group's MACs
OP_KERNEL(wg_s8_n192_a1, 256) { wn<96, 4, 0>(a); }
OP_KERNEL(wgli_redux_n192, 256) { wn<96, 4, 1>(a); }
OP_KERNEL(wgmix_imad_r256_n192, 256) { wn<96, 4, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_n192, 256) { wn<96, 4, 3>(a); }
OP_KERNEL(wg_s8_n32_a1, 256) { wn<16, 32, 0>(a); }
OP_KERNEL(wgli_redux_n32, 256) { wn<16, 32, 1>(a); }
OP_KERNEL(wgmix_imad_r256_n32, 256) { wn<16, 32, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_n32, 256) { wn<16, 32, 3>(a); }
OP_KERNEL(wg_s8_n16_a1, 256) { wn<8, 64, 0>(a); }
OP_KERNEL(wgli_redux_n16, 256) { wn<8, 64, 1>(a); }
OP_KERNEL(wgmix_imad_r256_n16, 256) { wn<8, 64, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_n16, 256) { wn<8, 64, 3>(a); }
OP_KERNEL(wg_s8_n8_a1, 256) { wn<4, 128, 0>(a); }
OP_KERNEL(wgli_redux_n8, 256) { wn<4, 128, 1>(a); }
OP_KERNEL(wgmix_imad_r256_n8, 256) { wn<4, 128, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_n8, 256) { wn<4, 128, 3>(a); }
OP_KERNEL(imma16816_a1, 256) { im<0, 1>(a); }
OP_KERNEL(wgli_redux_im16, 256) { im<1, 1>(a); }
OP_KERNEL(wgmix_imad_r256_im16, 256) { im<2, 1>(a); }
OP_KERNEL(wgmix_lop3_r256_im16, 256) { im<3, 1>(a); }
OP_KERNEL(bmma_a1, 256) { im<0, 2>(a); }
OP_KERNEL(wgli_redux_bm, 256) { im<1, 2>(a); }
OP_KERNEL(wgmix_imad_r256_bm, 256) { im<2, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_bm, 256) { im<3, 2>(a); }
OP_KERNEL(wg_f16_n256_a1, 256) { wf<0, 8, 0>(a); }
OP_KERNEL(wgli_redux_f16, 256) { wf<0, 8, 1>(a); }
OP_KERNEL(wgmix_imad_r256_f16, 256) { wf<0, 8, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_f16, 256) { wf<0, 8, 3>(a); }
OP_KERNEL(wg_bf16_n256_a1, 256) { wf<1, 8, 0>(a); }
OP_KERNEL(wgli_redux_bf, 256) { wf<1, 8, 1>(a); }
OP_KERNEL(wgmix_imad_r256_bf, 256) { wf<1, 8, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_bf, 256) { wf<1, 8, 3>(a); }
OP_KERNEL(wg_tf32_n256_a1, 256) { wf<2, 16, 0>(a); }
OP_KERNEL(wgli_redux_tf, 256) { wf<2, 16, 1>(a); }
OP_KERNEL(wgmix_imad_r256_tf, 256) { wf<2, 16, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_tf, 256) { wf<2, 16, 3>(a); }
OP_KERNEL(wg_e4m3_n256_a1, 256) { wf<3, 4, 0>(a); }
OP_KERNEL(wgli_redux_e4, 256) { wf<3, 4, 1>(a); }
OP_KERNEL(wgmix_imad_r256_e4, 256) { wf<3, 4, 2>(a); }
OP_KERNEL(wgmix_lop3_r256_e4, 256) { wf<3, 4, 3>(a); }

#define S8Q "IGMMA.64x64x32.S8.S8*=16"
#define IM "IMMA.16832.S8.S8=128"
#define S8_192 "IGMMA.64x192x32.S8.S8*=4"
#define S8_32 "IGMMA.64x32x32.S8.S8*=32"
#define S8_16 "IGMMA.64x16x32.S8.S8*=64"
#define S8_8 "IGMMA.64x8x32.S8.S8*=128"
#define IM16 "IMMA.16816.S8.S8*=256"
#define BM "BMMA.168256*=128"
#define F16 "HGMMA.64x256x16.F32*=8"
#define BF "HGMMA.64x256x16.F32.BF16*=8"
#define TF "HGMMA.64x256x8.F32.TF32*=16"
#define E4 "QGMMA.64x256x32.F32.E4M3.E4M3*=4"
static const OpProbe PROBES[] = {
    P(wg_s8_n256_a1, S8),
    P(wgli_redux, S8 LI),
    P(wgmix_imad_r256, S8 ",IMAD=256"),
    P(wgmix_lop3_r256, S8 ",LOP3.LUT=256"),
    P(wg_s8_n128_a1, S8H),
    P(wgli_redux_n128, S8H LI),
    P(wgmix_imad_r256_n128, S8H ",IMAD=256"),
    P(wgmix_lop3_r256_n128, S8H ",LOP3.LUT=256"),
    P(wg_s8_n64_a1, S8Q),
    P(wgli_redux_n64, S8Q LI),
    P(wgmix_imad_r256_n64, S8Q ",IMAD=256"),
    P(wgmix_lop3_r256_n64, S8Q ",LOP3.LUT=256"),
    P(imma_s8_a1, IM),
    P(wgli_redux_im, IM LI),
    P(wgmix_imad_r256_im, IM ",IMAD=256"),
    P(wgmix_lop3_r256_im, IM ",LOP3.LUT=256"),
    P(wg_s8_n192_a1, S8_192),
    P(wgli_redux_n192, S8_192 LI),
    P(wgmix_imad_r256_n192, S8_192 ",IMAD=256"),
    P(wgmix_lop3_r256_n192, S8_192 ",LOP3.LUT=256"),
    P(wg_s8_n32_a1, S8_32),
    P(wgli_redux_n32, S8_32 LI),
    P(wgmix_imad_r256_n32, S8_32 ",IMAD=256"),
    P(wgmix_lop3_r256_n32, S8_32 ",LOP3.LUT=256"),
    P(wg_s8_n16_a1, S8_16),
    P(wgli_redux_n16, S8_16 LI),
    P(wgmix_imad_r256_n16, S8_16 ",IMAD=256"),
    P(wgmix_lop3_r256_n16, S8_16 ",LOP3.LUT=256"),
    P(wg_s8_n8_a1, S8_8),
    P(wgli_redux_n8, S8_8 LI),
    P(wgmix_imad_r256_n8, S8_8 ",IMAD=256"),
    P(wgmix_lop3_r256_n8, S8_8 ",LOP3.LUT=256"),
    P(imma16816_a1, IM16),
    P(wgli_redux_im16, IM16 LI),
    P(wgmix_imad_r256_im16, IM16 ",IMAD=256"),
    P(wgmix_lop3_r256_im16, IM16 ",LOP3.LUT=256"),
    P(bmma_a1, BM),
    P(wgli_redux_bm, BM LI),
    P(wgmix_imad_r256_bm, BM ",IMAD=256"),
    P(wgmix_lop3_r256_bm, BM ",LOP3.LUT=256"),
    P(wg_f16_n256_a1, F16),
    P(wgli_redux_f16, F16 LI),
    P(wgmix_imad_r256_f16, F16 ",IMAD=256"),
    P(wgmix_lop3_r256_f16, F16 ",LOP3.LUT=256"),
    P(wg_bf16_n256_a1, BF),
    P(wgli_redux_bf, BF LI),
    P(wgmix_imad_r256_bf, BF ",IMAD=256"),
    P(wgmix_lop3_r256_bf, BF ",LOP3.LUT=256"),
    P(wg_tf32_n256_a1, TF),
    P(wgli_redux_tf, TF LI),
    P(wgmix_imad_r256_tf, TF ",IMAD=256"),
    P(wgmix_lop3_r256_tf, TF ",LOP3.LUT=256"),
    P(wg_e4m3_n256_a1, E4),
    P(wgli_redux_e4, E4 LI),
    P(wgmix_imad_r256_e4, E4 ",IMAD=256"),
    P(wgmix_lop3_r256_e4, E4 ",LOP3.LUT=256"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
