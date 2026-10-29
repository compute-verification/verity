// The census's unpriced sm_90a tensor class, DMMA (mma.sync f64: no H200 DS_tensor row lists f64), timed alone at
// cuBLAS's two heaviest shapes and beside a saturated wgmma s8. This is
// The DMMA-SHARES-TENSOR probe on the H200: unpriced.cu's B200 joint with wgmma for tcgen05.
// Built and gated by ../opcode_probe/probe.py build --arch sm_90a; rates.sh with ARCH=sm_90a is the job.
//   s_dmma, s_dmma_w8, s_imma   unpriced.cuh's: DMMA.8x8x4 (256 MACs a warp instruction), IMMA.16832 (4096)
//   s_dmma16, s_dmma16_w8       DMMA.16x8x16 (m16n8k16 f64, 2048 MACs a warp instruction), 8 chains a thread
//   wg_i8                       warps 0-7 of 16 (2 warpgroups) issue 4 wgmma m64n128k32 s8 a commit group: t's unit
//   wg256                       sm90_async.cu's saturating form (m64n256k32, 256 threads): wg_i8 should match its MACs
//   dmma[16]U_wg, wgU_dmma[16]  one role fixed, the other until it ends: DMMA on warps 8-15 (2 a SMSP), as unpriced.cu
// t + s > 1 (t the wgmma rate over wg_i8's, s the DMMA rate over its _w8 alone) says DMMA runs beside the tensor pipe
// and adds its words to a saturated wgmma's.
#include "../opcode_probe/probe.cuh"
#include "../dc/dc.cuh"
#include "../dc/wgmma.cuh"
#include "unpriced.cuh"

// DMMA m16n8k16: a thread's four f64 accumulators a chain, eight A and four B operands shared by every chain.
template <int C>
struct D16 {
  double c[C][4];
  double fa[8], fb[4];
  UP_DEV explicit D16(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < C; i++)
#pragma unroll
      for (int j = 0; j < 4; j++) c[i][j] = (double)((tid * 0x9e3779b9u + i * 0x85ebca6bu + j) & 0xfffu);
#pragma unroll
    for (int j = 0; j < 8; j++) fa[j] = (double)(int8_t)(tid * (37u + 2 * j));
#pragma unroll
    for (int j = 0; j < 4; j++) fb[j] = (double)(int8_t)(tid * (91u + 2 * j));
  }
  UP_DEV void op(int i) {
    const int s = i % C;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, "
        "{%12,%13,%14,%15}, {%0,%1,%2,%3};"
        : "+d"(c[s][0]), "+d"(c[s][1]), "+d"(c[s][2]), "+d"(c[s][3])
        : "d"(fa[0]), "d"(fa[1]), "d"(fa[2]), "d"(fa[3]), "d"(fa[4]), "d"(fa[5]), "d"(fa[6]), "d"(fa[7]), "d"(fb[0]),
          "d"(fb[1]), "d"(fb[2]), "d"(fb[3]));
  }
  UP_DEV uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < C; i++)
#pragma unroll
      for (int j = 0; j < 4; j++) x += (uint32_t)__double_as_longlong(c[i][j]) ^ (uint32_t)(__double_as_longlong(c[i][j]) >> 32);
    return x;
  }
};
template <int N>
UP_DEV void d16_alone(const OpArgs& a) {
  D16<8> c(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < N; q++) c.op(q);
  }
  OP_END
  OP_SINK(c.fold() ^ threadIdx.x)
}

// ---- wgmma s8 (sm90_async.cu's tiles: A 64 x 32 and B N x 32 bytes, K-major, no swizzle) ------------------------------
constexpr int KB = 32, AROWS = 64;
struct Tiles {
  uint64_t da, db;
};
template <int N>
UP_DEV Tiles wg_tiles(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS + N) * KB / 4; i += blockDim.x)
    w[i] = mode == 1 ? 0u : dc_mix32(i * 0x9E3779B9u + 0x1234567u);
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB))};
}
// Four wgmma into one accumulator a commit group: the first overwrites (scale_d 0), the rest accumulate.
UP_DEV void group4(int (&d)[64], const Tiles& t) {
  wgmma_fence();
  wgmma_s8_n128(d, t.da, t.db, 0);
  wgmma_s8_n128(d, t.da, t.db, 1);
  wgmma_s8_n128(d, t.da, t.db, 1);
  wgmma_s8_n128(d, t.da, t.db, 1);
  wgmma_commit();
}
UP_DEV void group4(int (&d)[128], const Tiles& t) {
  wgmma_fence();
  wgmma_s8_n256(d, t.da, t.db, 0);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_commit();
}
template <int N>
UP_DEV uint32_t fold_acc(const int (&d)[N]) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < N; i++) x ^= (uint32_t)d[i];
  return x;
}
// wgmma is warpgroup-collective, so the until role's stop must be one decision for the warpgroup's 4 warps: OR the
// flag over them on named barrier 1 + warpgroup (bar.red.or), where OP_UNTIL lets each warp read it on its own trip.
UP_DEV bool wg_any(int v, uint32_t wg) {
  uint32_t r;
  asm volatile(
      "{\n.reg .pred p, q;\nsetp.ne.s32 p, %1, 0;\nbar.red.or.pred q, %2, 128, p;\nselp.u32 %0, 1, 0, q;\n}\n"
      : "=r"(r)
      : "r"(v), "r"(1u + wg)
      : "memory");
  return r != 0;
}

OP_KERNEL(wg256, 256) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + 256) * KB];
  const Tiles t = wg_tiles<256>(sm, a.mode);
  int d[128] = {};
  OP_BEGIN
  OP_LOOP(it) {
    group4(d, t);
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ threadIdx.x)
}

// ROLE 0: wgmma alone (warps 0-7); 1: wgmma fixed, DMMA until; 2: DMMA fixed, wgmma until. S16: DMMA m16n8k16, else
// m8n8k4. Each role's registers live in its own branch, so the block fits 128 a thread.
template <int ROLE, bool S16>
UP_DEV void dmma_wg(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + 128) * KB];
  const Tiles t = wg_tiles<128>(sm, a.mode);
  const uint32_t w = warp_u();
  uint32_t x = 0;
  OP_BEGIN_STOP
  if (w < 8) {
    int d[64] = {};
    if constexpr (ROLE == 2) {
      int n = 0;
#pragma unroll 1
      for (; !wg_any(*(volatile int*)&op_stop_, w >> 2); ++n) {
        group4(d, t);
        wgmma_wait<0>();
      }
      OP_TRIPS(n)
    } else {
      OP_LOOP(it) {
        group4(d, t);
        wgmma_wait<0>();
      }
      if constexpr (ROLE == 1) {
        OP_DONE
      }
    }
    wgmma_hold(d);
    x = fold_acc(d);
  } else {
    if constexpr (ROLE != 0) {
      if constexpr (S16) {
        D16<8> c(threadIdx.x ^ (uint32_t)a.mode);
        auto dmma8 = [&] {
#pragma unroll
          for (int q = 0; q < 8; q++) c.op(q);
        };
        if constexpr (ROLE == 1) {
          OP_UNTIL(n) { dmma8(); }
          OP_TRIPS(n)
        } else {
          OP_LOOP(it) { dmma8(); }
          OP_DONE
        }
        x = c.fold();
      } else {
        Chains c(threadIdx.x ^ (uint32_t)a.mode);
        auto dmma8 = [&] {
#pragma unroll
          for (int q = 0; q < 8; q++) c.op<U_DMMA>(q);
        };
        if constexpr (ROLE == 1) {
          OP_UNTIL(n) { dmma8(); }
          OP_TRIPS(n)
        } else {
          OP_LOOP(it) { dmma8(); }
          OP_DONE
        }
        x = c.fold<U_DMMA>();
      }
    }
  }
  OP_END_STOP
  OP_SINK(x ^ threadIdx.x)
}

OP_KERNEL(s_dmma, 1024) { s_alone<U_DMMA, 8>(a); }
OP_KERNEL(s_dmma_w8, 256) { s_alone<U_DMMA, 8>(a); }
OP_KERNEL(s_imma, 1024) { s_alone<U_IMMA, 8>(a); }
OP_KERNEL(s_dmma16, 512) { d16_alone<8>(a); }
OP_KERNEL(s_dmma16_w8, 256) { d16_alone<8>(a); }
OP_KERNEL(wg_i8, 512) { dmma_wg<0, false>(a); }
OP_KERNEL(dmmaU_wg, 512) { dmma_wg<1, false>(a); }
OP_KERNEL(wgU_dmma, 512) { dmma_wg<2, false>(a); }
OP_KERNEL(dmma16U_wg, 512) { dmma_wg<1, true>(a); }
OP_KERNEL(wgU_dmma16, 512) { dmma_wg<2, true>(a); }

#define WG "IGMMA.64x128x32*=4"
#define D4 "DMMA.8x8x4*=8"
#define D16S "DMMA.16x8x16*=8"
#define S(NAME, TPB, PAT, IT, NOTE) {.name = #NAME, .kernel = (const void*)NAME, .tpb = TPB, .expect = PAT, .iters = IT, .note = NOTE}
#define J(NAME, PAT, NOTE, U) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = 512, .expect = PAT, .warps = "8;8", .iters = 2048, .note = NOTE, .until = U}
static const OpProbe PROBES[] = {
    S(s_dmma, 1024, D4, 256, "f64 MACs = 256 x rate; exact for int8 x int8 sums"),
    S(s_dmma_w8, 256, D4, 256, "DMMA.8x8x4 alone at the joint's 8 warps: s's unit"),
    S(s_imma, 1024, "IMMA*=8", 1024, "s8 MACs = 4096 x rate: the unit DMMA is read against"),
    S(s_dmma16, 512, D16S, 256, "f64 MACs = 2048 x rate"),
    S(s_dmma16_w8, 256, D16S, 256, "DMMA.16x8x16 alone at the joint's 8 warps: s's unit"),
    S(wg256, 256, "IGMMA.64x256x32*=4", 2048, "s8 MACs = 131072 x rate: the tensor pipe saturated (sm90_async)"),
    {.name = "wg_i8", .kernel = (const void*)wg_i8, .tpb = 512, .expect = WG, .warps = "8", .iters = 2048,
     .note = "s8 MACs = 65536 x rate, alone in the joint's block: t's unit"},
    J(dmmaU_wg, WG ";" D4, "wgmma fixed, DMMA.8x8x4 until", 2),
    J(wgU_dmma, WG ";" D4, "DMMA.8x8x4 fixed, wgmma until", 1),
    J(dmma16U_wg, WG ";" D16S, "wgmma fixed, DMMA.16x8x16 until", 2),
    J(wgU_dmma16, WG ";" D16S, "DMMA.16x8x16 fixed, wgmma until", 1),
};
#undef S
#undef J

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
