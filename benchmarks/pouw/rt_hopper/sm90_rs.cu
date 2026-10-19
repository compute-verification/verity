// RT-HOPPER-1 on the H200 (sm_90a): does a wgmma form other than the honest's run faster beside the LI fold? The census
// prices IGMMA_S8_N256_RS (A from registers) and IGMMA_U8S8_N256 on DC-90's joint rows JI90_0/JI90_1 at a lane
// instruction's default, 64x under the honest SS form's measured lines, so the model lets a cheater run either at the
// issue bound. These loops are ../dc/probes/sm90_wgmma.cu's, templated on the form: each form alone (wait_group 1), the
// honest LI-CP-W loop (wgli_redux: 4 wgmma, LI's 256 LOP3 + 256 IMAD, 4 checkpoints of 4 LOP3 and one redux.sync.xor,
// then wait), and the interleaved IMAD and LOP3 points. The RS form's A fragment is loop-invariant, the cheater's best
// case (no delivery). Built and gated by ../opcode_probe/probe.py; rs.sh runs it, rs.py reads it.
#include "../opcode_probe/probe.cuh"
#include "../dc/dc.cuh"
#include "../dc/probes/lanes.cuh"
#include "../dc/wgmma.cuh"

constexpr int KB = 32, AROWS = 64, BROWS = 256;

struct Tiles {
  uint64_t da, db;
};
DC_DEV Tiles fill_tiles(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x)
    w[i] = mode == 1 ? 0u : dc_mix32(i * 0x9E3779B9u + 0x1234567u);
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB))};
}

struct F_ss {
  DC_DEV static void mma(int (&d)[128], const Tiles& t, const uint32_t (&)[4], int sd) { wgmma_s8_n256(d, t.da, t.db, sd); }
};
struct F_rs {
  DC_DEV static void mma(int (&d)[128], const Tiles& t, const uint32_t (&a)[4], int sd) { wgmma_s8_n256_rs(d, a, t.db, sd); }
};
struct F_u8 {
  DC_DEV static void mma(int (&d)[128], const Tiles& t, const uint32_t (&)[4], int sd) { wgmma_u8s8_n256(d, t.da, t.db, sd); }
};

template <class F>
DC_DEV void group4(int (&d)[128], const Tiles& t, const uint32_t (&a)[4]) {
  wgmma_fence();
  F::mma(d, t, a, 0);
  F::mma(d, t, a, 1);
  F::mma(d, t, a, 1);
  F::mma(d, t, a, 1);
  wgmma_commit();
}

DC_DEV uint32_t fold_acc(const int (&d)[128]) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 128; i++) x ^= (uint32_t)d[i];
  return x;
}

#define SETUP                                                                                                   \
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];                                            \
  const Tiles t = fill_tiles(sm, a.mode);                                                                       \
  const uint32_t af[4] = {(uint32_t)t.da ^ threadIdx.x, (uint32_t)(t.da >> 32), (uint32_t)t.db, 0x01020304u}; \
  int d[128] = {};

template <class F>
DC_DEV void wg_alone(const OpArgs& a) {
  SETUP
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t, af);
    wgmma_wait<1>();
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ threadIdx.x)
}

template <class F>
DC_DEV void wg_li(const OpArgs& a) {
  SETUP
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t, af);
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
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}

template <class F, int X, int R>
DC_DEV void wg_mix(const OpArgs& a) {
  SETUP
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t, af);
    lane_ops<X, R>(l);
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}

OP_KERNEL(wg_s8_n256_a1, 256) { wg_alone<F_ss>(a); }
OP_KERNEL(wg_s8_n256_rs_a1, 256) { wg_alone<F_rs>(a); }
OP_KERNEL(wg_u8s8_n256_a1, 256) { wg_alone<F_u8>(a); }
OP_KERNEL(wgli_redux, 256) { wg_li<F_ss>(a); }
OP_KERNEL(wgli_redux_rs, 256) { wg_li<F_rs>(a); }
OP_KERNEL(wgli_redux_u8, 256) { wg_li<F_u8>(a); }
OP_KERNEL(wgmix_imad_r256, 256) { wg_mix<F_ss, L_IMAD, 256>(a); }
OP_KERNEL(wgmix_imad_r256_rs, 256) { wg_mix<F_rs, L_IMAD, 256>(a); }
OP_KERNEL(wgmix_lop3_r256, 256) { wg_mix<F_ss, L_LOP3, 256>(a); }
OP_KERNEL(wgmix_lop3_r256_rs, 256) { wg_mix<F_rs, L_LOP3, 256>(a); }

#define S8 "IGMMA.64x256x32.S8.S8*=4"
#define U8 "IGMMA.64x256x32.U8.S8*=4"
#define LI ",LOP3.LUT=272,IMAD=256,REDUX*=4"
#define P(NAME, PAT) {.name = #NAME, .kernel = (const void*)NAME, .tpb = 256, .expect = PAT, .iters = 4096}
static const OpProbe PROBES[] = {
    P(wg_s8_n256_a1, S8),
    P(wg_s8_n256_rs_a1, S8),
    P(wg_u8s8_n256_a1, U8),
    P(wgli_redux, S8 LI),
    P(wgli_redux_rs, S8 LI),
    P(wgli_redux_u8, U8 LI),
    P(wgmix_imad_r256, S8 ",IMAD=256"),
    P(wgmix_imad_r256_rs, S8 ",IMAD=256"),
    P(wgmix_lop3_r256, S8 ",LOP3.LUT=256"),
    P(wgmix_lop3_r256_rs, S8 ",LOP3.LUT=256"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
