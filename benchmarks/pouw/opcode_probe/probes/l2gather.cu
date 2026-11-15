// The sustained L2 random-row gather rate, which machine_model/data/table.toml records beside `l2` (3.85 TB/s). It is
// no census row and doesn't price the table route: a rate at random rows rests on the cheater's access pattern, which
// the cheater controls, so no probe here writes a census line.
// Each lane gathers one row of W bytes from an L2-resident table at a fresh pseudo-random index (independent each op, so
// no coalescing and no reuse), via one load of 64/128/256 bits. The table bytes are
// set at runtime from a.mode (its log2), swept 1 MB..512 MB, so the rate-vs-size curve shows the L1/L2 cliff and the DRAM
// floor past the L2. Each path runs the gather alone (gl_/gt_/gn_, many warps to saturate L2) and beside the saturated
// int8 IMMA honest step with the duo fold (gh_, OP_UNTIL), so the best sustained bytes/SM/clk and the MMA rate kept are
// both read. ILP is the loads a trip; several warp counts.
//   gl_w<W>[_<V>]  the LDG path at W bytes a lane, skipping L1 (ld.global.cg: LDG.E.64.STRONG.GPU, LDG.E.128.STRONG.GPU,
//                  LDG.E.ENL2.256.STRONG.GPU); V names an ILP (i4/i16) or warp (w8/w12) variant, else ILP 8 and 16
//                  warps alone / 4 side warps beside
//   gt_w4          the texture path (tex1Dfetch<u32>, 4 B a lane: TLD.LZ)
//   gn_w16         the read-only path (__ldg of an int4, 16 B a lane: LDG.E.128.CONSTANT)
// The index is a fresh xorshift word a (lane, trip, op), independent of the loaded data (max bandwidth, a cheater's best
// case); the loaded words fold into the sink. eps1.py's `l2gather` reads each loop's load count from its SASS, the bytes
// a SM a clock and TB/s, and the best sustained.
#include "../probe.cuh"
#include "honest.cuh"

constexpr int HW = 16;  // honest warps beside the gather

enum { P_LDG, P_TEX, P_RO };

// one fresh pseudo-random index a (lane, trip, op): xorshift of a seed that no two ops share
__device__ __forceinline__ uint32_t idx_of(uint32_t gtid, uint32_t seq) {
  uint32_t h = gtid * 0x9E3779B1u + seq * 0x85EBCA77u + 0xC2B2AE3Du;
  h ^= h << 13;
  h ^= h >> 17;
  h ^= h << 5;
  return h;
}

// W bytes from L2 at row `row & (rows-1)` of a table of `rows` rows, folded to one word; path P, bypassing L1 on the LDG path
template <int W, int P>
__device__ __forceinline__ uint32_t gather(const OpArgs& a, uint32_t row, uint32_t rows) {
  const uint32_t r = row & (rows - 1);
  if constexpr (P == P_TEX) return tex1Dfetch<uint32_t>(a.tex1, (int)r);
  const unsigned char* p = a.g + (size_t)r * W;
  if constexpr (P == P_RO) {
    const int4 v = __ldg((const int4*)p);
    return (uint32_t)v.x ^ v.y ^ v.z ^ v.w;
  }
  if constexpr (W == 8) {
    uint32_t x, y;
    asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "l"(p));
    return x ^ y;
  }
  if constexpr (W == 16) {
    uint32_t x, y, z, w;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(p));
    return x ^ y ^ z ^ w;
  }
  if constexpr (W == 32) {
    uint32_t v[8];
    asm volatile("ld.global.cg.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                 : "l"(p));
    return v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[4] ^ v[5] ^ v[6] ^ v[7];
  }
  return 0;
}

// a gather role: this warp is side warp `sw` of `nsw`; R independent loads a trip
template <int W, int P, int R>
struct Gather {
  const OpArgs* ap;
  uint32_t gtid, acc, rows;
  __device__ Gather(const OpArgs& a, int sw, int nsw) : ap(&a) {
    const uint32_t lane = threadIdx.x & 31;
    gtid = (blockIdx.x * nsw + sw) * 32 + lane;
    acc = gtid;
    rows = (uint32_t)((1ull << a.mode) / (P == P_TEX ? 4u : (unsigned)W));  // a power of two: the table's rows
  }
  __device__ __forceinline__ void trip(uint32_t t) {
#pragma unroll
    for (int q = 0; q < R; q++) acc ^= gather<W, P>(*ap, idx_of(gtid, t * R + q), rows);
  }
};

template <int W, int P, int R, bool SIDE>
__device__ __forceinline__ void run(const OpArgs& a, int side_warps) {
  const int w = threadIdx.x >> 5;
  Gather<W, P, R> g(a, SIDE ? w - HW : w, side_warps);
  if constexpr (SIDE) {
    Honest h(a.mode);
    OP_BEGIN_STOP
    if (w < HW) {
      OP_LOOP(it) { h.step(); }
      OP_DONE
    } else {
      OP_UNTIL(n) { g.trip((uint32_t)n); }
      OP_TRIPS(n)
    }
    OP_END_STOP
    OP_SINK(h.fold() ^ g.acc)
  } else {
    OP_BEGIN
    OP_LOOP(it) { g.trip((uint32_t)it); }
    OP_END
    OP_SINK(g.acc)
  }
}

constexpr int ALW = 16, SW = 4;  // alone: 16 warps a block; beside: 16 honest + 4 side (one a SMSP)

#define GL(nm, W, P, R, WARPS) \
  OP_KERNEL(nm, (WARPS) * 32) { run<W, P, R, false>(a, WARPS); }
#define GH(nm, W, P, R, SIDEW) \
  OP_KERNEL(nm, (HW + SIDEW) * 32) { run<W, P, R, true>(a, SIDEW); }

GL(gl_w8, 8, P_LDG, 8, ALW)
GL(gl_w16, 16, P_LDG, 8, ALW)
GL(gl_w32, 32, P_LDG, 8, ALW)
GL(gl_w32_i4, 32, P_LDG, 4, ALW)
GL(gl_w32_i16, 32, P_LDG, 16, ALW)
GL(gl_w32_w8, 32, P_LDG, 8, 8)
GL(gl_w32_w12, 32, P_LDG, 8, 12)
GL(gt_w4, 4, P_TEX, 8, ALW)
GL(gn_w16, 16, P_RO, 8, ALW)
GH(gh_w8, 8, P_LDG, 8, SW)
GH(gh_w16, 16, P_LDG, 8, SW)
GH(gh_w32, 32, P_LDG, 8, SW)
GH(gh_w32_w8, 32, P_LDG, 8, 8)
GH(gt_h4, 4, P_TEX, 8, SW)

// honest alone (16 warps, the side warps idle): the IMMA rate the beside probes' "MMA kept" is a fraction of
OP_KERNEL(honest, (HW + SW) * 32) {
  Honest h(a.mode);
  OP_BEGIN
  if ((threadIdx.x >> 5) < HW) { OP_LOOP(it) { h.step(); } }
  OP_END
  OP_SINK(h.fold())
}

#define IMMA "IMMA.16832.S8.S8"
#define LDG8 "LDG.E.64.STRONG.GPU"
#define LDG16 "LDG.E.128.STRONG.GPU"
#define LDG32 "LDG.E.ENL2.256.STRONG.GPU"
#define LDGRO "LDG.E.128.CONSTANT"
#define TLD "TLD.LZ"
// words a warp instruction = (bytes a lane / 4) * 32 lanes
#define WORDS(W) ((W) / 4.0 * 32)

// alone: one gather loop of R loads. beside: IMMA=8 and R loads, a joint point. Both rows.jsonl only.
#define AL(KN, W, MB, LOG, WARPS, PAT)                                                                              \
  {.name = #KN "_s" #MB, .kernel = (const void*)KN, .tpb = (WARPS) * 32,                                            \
   .expect = PAT "=8", .counted = PAT, .words = WORDS(W), .mode = LOG, .iters = 4096,                               \
   .note = "s" #MB, .fn = #KN},
#define BE(KN, W, MB, LOG, SIDEW, PAT)                                                                              \
  {.name = #KN "_s" #MB, .kernel = (const void*)KN, .tpb = (HW + SIDEW) * 32,                                       \
   .expect = IMMA "=8;" PAT "=8", .counted = PAT, .warps = "16;" #SIDEW, .words = WORDS(W), .mode = LOG,            \
   .iters = 4096, .note = "s" #MB " beside", .fn = #KN, .until = 2},

static const OpProbe PROBES[] = {
    // honest alone: the IMMA rate the beside probes' MMA kept is a fraction of
    {.name = "honest", .kernel = (const void*)honest, .tpb = (HW + SW) * 32, .expect = IMMA "=8", .counted = IMMA,
     .warps = "16", .iters = 4096, .note = "honest alone (16 warps)"},
    // alone, LDG, width 8/16/32, ILP 8, 16 warps, over 1 MB..512 MB
    AL(gl_w8, 8, 1, 20, ALW, LDG8) AL(gl_w8, 8, 16, 24, ALW, LDG8)
    AL(gl_w8, 8, 64, 26, ALW, LDG8) AL(gl_w8, 8, 128, 27, ALW, LDG8)
    AL(gl_w8, 8, 256, 28, ALW, LDG8) AL(gl_w8, 8, 512, 29, ALW, LDG8)
    AL(gl_w16, 16, 1, 20, ALW, LDG16) AL(gl_w16, 16, 16, 24, ALW, LDG16)
    AL(gl_w16, 16, 64, 26, ALW, LDG16) AL(gl_w16, 16, 128, 27, ALW, LDG16)
    AL(gl_w16, 16, 256, 28, ALW, LDG16) AL(gl_w16, 16, 512, 29, ALW, LDG16)
    AL(gl_w32, 32, 1, 20, ALW, LDG32) AL(gl_w32, 32, 16, 24, ALW, LDG32)
    AL(gl_w32, 32, 64, 26, ALW, LDG32) AL(gl_w32, 32, 128, 27, ALW, LDG32)
    AL(gl_w32, 32, 256, 28, ALW, LDG32) AL(gl_w32, 32, 512, 29, ALW, LDG32)
    // beside honest, LDG, width 8/16/32, ILP 8, 4 side warps (one a SMSP)
    BE(gh_w8, 8, 1, 20, 4, LDG8) BE(gh_w8, 8, 16, 24, 4, LDG8) BE(gh_w8, 8, 64, 26, 4, LDG8)
    BE(gh_w8, 8, 128, 27, 4, LDG8) BE(gh_w8, 8, 256, 28, 4, LDG8) BE(gh_w8, 8, 512, 29, 4, LDG8)
    BE(gh_w16, 16, 1, 20, 4, LDG16) BE(gh_w16, 16, 16, 24, 4, LDG16) BE(gh_w16, 16, 64, 26, 4, LDG16)
    BE(gh_w16, 16, 128, 27, 4, LDG16) BE(gh_w16, 16, 256, 28, 4, LDG16) BE(gh_w16, 16, 512, 29, 4, LDG16)
    BE(gh_w32, 32, 1, 20, 4, LDG32) BE(gh_w32, 32, 16, 24, 4, LDG32) BE(gh_w32, 32, 64, 26, 4, LDG32)
    BE(gh_w32, 32, 128, 27, 4, LDG32) BE(gh_w32, 32, 256, 28, 4, LDG32) BE(gh_w32, 32, 512, 29, 4, LDG32)
    // ILP and warp variants at width 32, the L2 and beyond-L2 sizes
    {.name = "gl_w32_i4_s128", .kernel = (const void*)gl_w32_i4, .tpb = ALW * 32, .expect = LDG32 "=4",
     .counted = LDG32, .words = WORDS(32), .mode = 27, .iters = 4096, .note = "s128 ilp4", .fn = "gl_w32_i4"},
    {.name = "gl_w32_i4_s256", .kernel = (const void*)gl_w32_i4, .tpb = ALW * 32, .expect = LDG32 "=4",
     .counted = LDG32, .words = WORDS(32), .mode = 28, .iters = 4096, .note = "s256 ilp4", .fn = "gl_w32_i4"},
    {.name = "gl_w32_i16_s128", .kernel = (const void*)gl_w32_i16, .tpb = ALW * 32, .expect = LDG32 "=16",
     .counted = LDG32, .words = WORDS(32), .mode = 27, .iters = 2048, .note = "s128 ilp16", .fn = "gl_w32_i16"},
    {.name = "gl_w32_i16_s256", .kernel = (const void*)gl_w32_i16, .tpb = ALW * 32, .expect = LDG32 "=16",
     .counted = LDG32, .words = WORDS(32), .mode = 28, .iters = 2048, .note = "s256 ilp16", .fn = "gl_w32_i16"},
    {.name = "gl_w32_w8_s128", .kernel = (const void*)gl_w32_w8, .tpb = 8 * 32, .expect = LDG32 "=8",
     .counted = LDG32, .words = WORDS(32), .mode = 27, .iters = 4096, .note = "s128 warps8", .fn = "gl_w32_w8"},
    {.name = "gl_w32_w12_s128", .kernel = (const void*)gl_w32_w12, .tpb = 12 * 32, .expect = LDG32 "=8",
     .counted = LDG32, .words = WORDS(32), .mode = 27, .iters = 4096, .note = "s128 warps12", .fn = "gl_w32_w12"},
    {.name = "gh_w32_w8_s128", .kernel = (const void*)gh_w32_w8, .tpb = (HW + 8) * 32,
     .expect = IMMA "=8;" LDG32 "=8", .counted = LDG32, .warps = "16;8", .words = WORDS(32), .mode = 27, .iters = 4096,
     .note = "s128 beside warps8", .fn = "gh_w32_w8", .until = 2},
    // the texture (4 B) and read-only (16 B) paths, the L2 and beyond-L2 sizes
    AL(gt_w4, 4, 64, 26, ALW, TLD) AL(gt_w4, 4, 128, 27, ALW, TLD)
    BE(gt_h4, 4, 128, 27, 4, TLD)
    AL(gn_w16, 16, 128, 27, ALW, LDGRO) AL(gn_w16, 16, 256, 28, ALW, LDGRO)
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
