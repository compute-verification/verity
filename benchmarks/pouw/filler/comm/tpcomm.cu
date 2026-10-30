// Idle tensor time in tensor-parallel decode over PCIe, and the DRAM-free filler inside it.
//
// A proxy of one Llama-3.1-70B decode step at tensor parallelism TP on one node, one process a rank with every leased GPU
// visible (rank r runs on device r), as vLLM's workers run. Per layer, in vLLM's order: norm + quantize, qkv (column-parallel,
// n = 10240/TP), RoPE + cache write, attention over the batch's context (the rank's 8/TP KV heads), quantize, o (row-parallel,
// k = 8192/TP), NCCL all-reduce of the B x 8192 BF16 hidden state, residual + norm + quantize, gate_up (n = 57344/TP),
// SiLU-and-mul + quantize, down (k = 28672/TP), all-reduce; one all-reduce for the vocab-parallel embedding first, and the
// lm_head shard (n = 128256/TP) with the all-gather of its BF16 logits last. The linears are int8 cuBLASLt GEMMs (dcap.cu's
// lt_make: the fastest of the heuristic's top 16 whose output matches the host), distinct weights a layer; attention is
// dcap.cu's int8 decode attention (QK and PV, dp4a, memory-bound) over B/TP sequences of dcap's 8 KV heads, which reads the
// same KV bytes as B sequences of the rank's 8/TP heads; the small kernels (norms, RoPE, quantize, SiLU) are one
// elementwise kernel each that reads and writes their bytes. Each arm is one CUDA graph a rank, NCCL captured inside it.
//
// Arms (each rank, in lockstep through the collectives): base (the step), nocomm (the step without its collectives), comm
// (the collectives alone), phase (the step with a filler beside every collective). Served work is the int8 MACs the step's
// linears and attention execute at 70B's shapes (QK and PV at 64/TP query heads); capacity is 1024 MACs/SM/clk at the SM
// clock NVML reads during each rep. Clocks are read, never set.
//
// The phase filler is pfill-k (pfill.cuh built with PFILL_BITS): Theorem K's
// tile, {0,1} bits on both operands of the s8 IMMA m16n8k32, each k32 s32 word folded unscaled into the ram receipt as
// it leaves the MMA. It is launched on a side branch of the graph beside each collective, one block of KF_G tile groups
// (PF_W warps each, pfill's tile loop) on each of SMs - TPC_RESERVE SMs, so that NCCL's blocks find free SMs. Group gid of
// collective `slot` runs up to TPC_TILES tiles (TPC_TILES_G at the closing all-gather) with the public ids
// (epoch·nslots + slot)·groups·maxcap + gid·maxcap + i, and stops when the kernel after the collective raises the
// collective's flag. Each warp polls a read of the flag issued one poll earlier every TPC_POLL B fragments (a multiple of
// KF_PJ; 0: only between tiles; -1: never, fixed work) and stops there, mid-tile. Since every word enters its chain at
// once, a warp's chains after q (fragment group, B fragment) pairs are the receipt of exactly those 4q MMAs, so a warp
// writes its receipt and q for every tile it started, and the host recomputes the warp's chains for q pairs
// (kf_ref_warp). A whole tile's eight warp receipts XOR to pfill's own tile receipt (dfill_ref of pfill.cuh).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a tpcomm.cu -lcublasLt -lnvidia-ml -ldl -lnccl
// dcap.cu's filler is pfill.cuh's, here its bits tile (PFILL_BITS); dcap.cu's phase-filler state takes another name, since
// `PFill` below is this file's.
#define PFILL_BITS
#define PFill DcapPFill
#define main dcap_main
#include "../dcap.cu"
#undef main
#undef PFill

#include <nccl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <map>

// The fold the TP filler path (tp_dfill_k) and the deferred pass commit their receipts with (the filler receipt
// break). Default duo (NC 2): it binds every word under Theorem K (c = 1 at φ 0.9), provisionally: the binding
// is proved for runs of one to four words (Lean Pouw.SecurityProofs.Receipt.DuoQuadNoKernel), and five or more words
// and the served receipt's projected form are open. -DTP_NC=4 is the ram
// fold, which binds a word PAIR only through Q = a0·d0 + d1, so it certifies half the words (round 8). At NC 4 one
// of the r9 XOR-family folds is built with -DPFILL_LI or -DPFILL_XIN13 (pfill.cuh's r9 folds); each is
// provisional and reads at Theorem K's proved half (0.45) until the select-charging lemma lands. -DPFILL_XIN
// (rotl 7) is retired (pfill.cuh #errors on it). pf_absorb and pf_fold read NC, so one macro picks the fold for every
// launch and every host reference.
#ifndef TP_NC
#define TP_NC 2
#endif
static_assert(TP_NC == 2 || TP_NC == 4, "TP_NC is the receipt fold: 2 (duo, binding proved for runs of one to four words) or 4 (ram; -DPFILL_LI/-DPFILL_XIN13 for r9's XOR-family)");
#if defined(PFILL_LI) || defined(PFILL_XIN13)
constexpr int TP_XIN = 1;  // an XOR-family r9 fold at NC 4 (li or xin13); reads at 0.45 like ram, labelled apart
#else
constexpr int TP_XIN = 0;
#endif

#define NK(x)                                                                                     \
  do {                                                                                            \
    ncclResult_t r_ = (x);                                                                        \
    if (r_ != ncclSuccess) {                                                                      \
      fprintf(stderr, "NCCL %s at %s:%d: %s\n", #x, __FILE__, __LINE__, ncclGetErrorString(r_)); \
      exit(2);                                                                                    \
    }                                                                                             \
  } while (0)

constexpr int T_LAYERS = 80, T_HID = 8192, T_QKV = 10240, T_FF = 28672, T_VOCAB = 128256, T_HQ = 64, T_HKV = 8, T_HD = 128;

// ---------------------------------------------------------------- kernels
// one of the step's small kernels: reads nin and writes nout 16-byte words; the first thread raises `flag` (the phase filler's
// stop for the collective just before it)
__global__ void glue_k(const int4* __restrict__ in, size_t nin, int4* __restrict__ out, size_t nout, int* flag) {
  if (flag && blockIdx.x == 0 && threadIdx.x == 0) *(volatile int*)flag = 1;
  const size_t st = (size_t)gridDim.x * blockDim.x, n = nin > nout ? nin : nout;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += st) {
    int4 v = i < nin ? in[i] : make_int4(0, 0, 0, 0);
    if (i < nout) out[i] = make_int4(v.x ^ (int)i, v.y, v.z, v.w);
  }
}
// the step's first kernel when a phase filler runs: every collective's flag down, the epoch up
__global__ void prolog_k(int* flags, int n, unsigned long long* epoch) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) flags[i] = 0;
  if (threadIdx.x == 0) *epoch += 1;
}
__device__ __forceinline__ int ld_flag(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
constexpr int KF_G = 2;   // tile groups a block: one block an SM, 2 x 8 warps at 128 registers fill the register file
constexpr int KF_PJ = 16;  // B fragments (64 MMAs a warp) between a warp's polls of the flag
constexpr int KF_PAIRS_J = PF_TN / 8;
constexpr int KF_H = PF_W / 2;            // warps of a group that run a tile's first SHA pass: one an SMSP
constexpr int KF_HB = PF_TN / 16 / KF_H;  // B's SHA blocks each of them makes (lanes 0 .. KF_HB - 1)
constexpr int KF_A0 = 8;                  // A fragments each makes for itself; for its upper partner, 4
static_assert(KF_HB + 4 + KF_A0 + 1 <= 32 && KF_A0 % 4 == 0 && KF_A0 >= 8, "the first pass's lanes: B, 4, KF_A0, key");
constexpr size_t KF_SMEM_GRP = (size_t)PF_TN * 32 + (size_t)PF_W * 32 * 16 * 4 + sizeof(PfKeys);  // B as s8 bytes, A's
                                                                                                  // 32 SHA blocks a warp
constexpr size_t KF_SMEM = KF_G * KF_SMEM_GRP;
static_assert(KF_SMEM_GRP % 16 == 0 && KF_SMEM <= 101376, "a group's B is 16-byte aligned; one block an SM");
// bar.sync / bar.red over one tile group's 32 PF_W threads (barrier 0 is __syncthreads')
__device__ __forceinline__ void kf_bar(int id) { asm volatile("bar.sync %0, %1;" ::"r"(id), "n"(32 * PF_W) : "memory"); }
__device__ __forceinline__ int kf_bar_or(int id, int v) {
  int r;
  asm volatile("{\n\t.reg .pred p, q;\n\tsetp.ne.s32 p, %1, 0;\n\tbar.red.or.pred q, %2, %3, p;\n\tselp.s32 %0, 1, 0, q;\n\t}"
               : "=r"(r)
               : "r"(v), "r"(id), "n"(32 * PF_W)
               : "memory");
  return r;
}
// Group gid = KF_G blockIdx + (warp / PF_W) of collective `slot` runs tiles base + i, i < tiles, base = (epoch nslots +
// slot) groups maxcap + gid maxcap: pfill_tiles' tile (PFILL-b1, ram receipt) warp for warp, except that a warp stops at a
// poll that reads the flag up. Records (slot, gid, i, w) hold warp w's receipt of tile i, the XOR over its lanes and chains
// of pf_fold(dseed(c0, c1, tile), pos, y) at the state it stopped in, and q, the (fragment group, B fragment) pairs it ran
// (4 MMAs each; KF_PAIRS_J S / 4 when whole). cnt[slot, gid] is the tiles the group started; every one but the last is
// whole. total is 2 u64 a warp of a collective (gw = (slot·groups + gid)·PF_W + w): total[2 gw] the pairs the warp ran,
// total[2 gw + 1] its group's whole-tile pairs once (w == 0, else 0). The host sums them (pf_total_sum). A store-only put
// can't accumulate, so each collective's launch overwrites its own slots: the sum is the last step's work over every
// collective (it must equal the records' kf_shape sums), not the sum over time_work's repeats.
//
// The body is tp_dfill_k, SCORER's filler path: a `__noinline__` device function whose name holds tp_dfill_k, so the
// scorer reads it as the proved fill (its SASS is only IMMA.16832.S8.S8 with C = RZ). kfill_k is its thin wrapper. The
// filler's four global writes are store-only `tp_dfill_k_put*` functions (their names hold tp_dfill_k too, so the scorer
// reads each as the filler and holds it to stores of its arguments alone, SCORER v5 `filler_paths`/`_put_bad`): the
// receipt and pair count a tile, the per-warp counters, and the per-group tile count. Because a put holds no reduction,
// the two per-warp counters are plain stores summed on the host, not atomics.

// receipt and pairs for tile i (lane 0): rec[ix] = h, qrec[ix] = q, its arguments
__device__ __noinline__ void tp_dfill_k_put(unsigned long long* __restrict__ rec, int* __restrict__ qrec, size_t ix,
                                            unsigned long long h, int q) {
  rec[ix] = h;
  qrec[ix] = q;
}
// the warp's per-launch counters as plain stores (lane 0): total[j] its pairs, total[j + 1] its group's whole-tile pairs
__device__ __noinline__ void tp_dfill_k_put_total(unsigned long long* __restrict__ total, size_t j,
                                                  unsigned long long mine, unsigned long long whole) {
  total[j] = mine;
  total[j + 1] = whole;
}
// the group's tiles started (group leader): cnt[i] = started, its argument
__device__ __noinline__ void tp_dfill_k_put_cnt(int* __restrict__ cnt, size_t i, int started) { cnt[i] = started; }

template <int NC>
__device__ __noinline__ void tp_dfill_k(uint64_t c0, uint64_t c1, const unsigned long long* __restrict__ epoch, int slot,
                                        int nslots, int tiles, int maxcap, int S, int poll, const int* flag,
                                        unsigned long long* __restrict__ rec, int* __restrict__ qrec,
                                        int* __restrict__ cnt, unsigned long long* total) {
  extern __shared__ __align__(16) uint8_t kf_smem[];
  constexpr int P = 32;
  constexpr uint64_t dom = PF_DOM01;
  const int grp = threadIdx.x / (32 * PF_W), bar = 1 + grp;
  uint8_t* Bs = kf_smem + grp * KF_SMEM_GRP;
  uint32_t(*Ab)[32 * 16] = reinterpret_cast<uint32_t(*)[32 * 16]>(Bs + PF_TN * 32);
  PfKeys* ks = reinterpret_cast<PfKeys*>(Bs + PF_TN * 32 + PF_W * 32 * 16 * 4);
  const int lane = threadIdx.x & 31, w = (threadIdx.x >> 5) % PF_W, g = lane >> 2, t = lane & 3;
  const unsigned long long ng = (unsigned long long)gridDim.x * KF_G, gid = (unsigned long long)blockIdx.x * KF_G + grp;
  const unsigned long long base = (*epoch * (unsigned long long)nslots + slot) * ng * maxcap + gid * maxcap;
  const size_t rbase = ((size_t)slot * ng + gid) * maxcap;
  const int full = S / 4 * KF_PAIRS_J;
  int fr = poll >= 0 ? ld_flag(flag) : 0;  // tested a poll later, so its latency overlaps MMAs
  int started = 0, qlast = full, left = poll;
  unsigned long long mine = 0;
  for (int i = 0; i < tiles; i++) {
    // group-uniform, and it orders the last tile's reads of Bs and Ab before they are rewritten
    if (kf_bar_or(bar, fr)) break;
    if (poll >= 0) fr = ld_flag(flag);
    const unsigned long long tile = base + i;
    started++;
    // The tile's first SHA pass runs on the group's lower KF_H warps alone, one a group on each SMSP, so an SM's start
    // is 2 passes deep instead of 4 (a pass 4 deep is about 12 us of a 46-240 us window). Lower warp w: lanes
    // 0 .. KF_HB - 1 hash B's blocks KF_HB w + lane, the next 4 its upper partner w + KF_H's A fragments 0..3, the next
    // KF_A0 its own fragments 0 .. KF_A0 - 1, the next the key block. Each warp then regenerates A itself (agen), the
    // upper ones at fragment 4 and the lower at KF_A0, so the two warps of a pair on an SMSP hash at different times.
    if (w < KF_H) {
      int wa = -1, fa = 0;
      uint64_t ctr = PF_KEY;
      if (lane < KF_HB) {
        ctr = PF_B + (uint64_t)(KF_HB * w + lane);
      } else if (lane < KF_HB + 4 + KF_A0) {
        wa = lane < KF_HB + 4 ? w + KF_H : w;
        fa = lane < KF_HB + 4 ? lane - KF_HB : lane - KF_HB - 4;
        ctr = (uint64_t)wa * S + fa;
      }
      if (lane <= KF_HB + 4 + KF_A0) {
        uint64_t st[8];
        pf_block(dom, c0, c1, tile, ctr, st);
        if (lane < KF_HB) {
          const int bb = KF_HB * w + lane;
#pragma unroll
          for (int nn = 0; nn < 16; nn++) {
            const uint32_t R = pf_row(st, nn);
            uint4* p = (uint4*)&Bs[(16 * bb + nn) * 32];
            p[0] = make_uint4(pf_01(R & 15), pf_01((R >> 16) & 15), pf_01((R >> 4) & 15), pf_01((R >> 20) & 15));
            p[1] = make_uint4(pf_01((R >> 8) & 15), pf_01((R >> 24) & 15), pf_01((R >> 12) & 15), pf_01(R >> 28));
          }
        } else if (wa >= 0) {
          uint4* p = (uint4*)&Ab[wa][fa * 16];
#pragma unroll
          for (int q = 0; q < 4; q++)
            p[q] = make_uint4(pf_row(st, 4 * q), pf_row(st, 4 * q + 1), pf_row(st, 4 * q + 2), pf_row(st, 4 * q + 3));
        } else if (w == 0) {
          *ks = pf_keys(st);
        }
      }
    }
    kf_bar(bar);
    const PfKeys key = *ks;
    auto agen = [&](int f0) {
      __syncwarp();
      if (f0 + lane < S) {
        uint64_t st[8];
        pf_block(dom, c0, c1, tile, (uint64_t)w * S + f0 + lane, st);
        uint4* p = (uint4*)&Ab[w][lane * 16];
#pragma unroll
        for (int q = 0; q < 4; q++)
          p[q] = make_uint4(pf_row(st, 4 * q), pf_row(st, 4 * q + 1), pf_row(st, 4 * q + 2), pf_row(st, 4 * q + 3));
      }
      __syncwarp();
    };
    uint32_t y[4], z[4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const uint32_t pos = (uint32_t)((w * 32 + lane) * 4 + c);
      y[c] = pf_y0(pos);
      z[c] = pf_z0(pos);
    }
    int f0 = 0, next = w < KF_H ? KF_A0 : 4;
    const uint8_t* bp = Bs + g * 32 + 8 * t;
    int q = 0;
    bool stop = false;
#pragma unroll 1
    for (int f = 0; f < S && !stop; f += 4) {
      if (f == next) {
        f0 = f;
        agen(f0);
        next = f + P;
      }
      uint32_t a[4][4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const uint32_t* r = &Ab[w][(f - f0 + c) * 16];
        a[c][0] = pf_01((r[g] >> (4 * t)) & 15);
        a[c][1] = pf_01((r[g + 8] >> (4 * t)) & 15);
        a[c][2] = pf_01((r[g] >> (16 + 4 * t)) & 15);
        a[c][3] = pf_01((r[g + 8] >> (16 + 4 * t)) & 15);
      }
#pragma unroll 1
      for (int j0 = 0; j0 < KF_PAIRS_J; j0 += KF_PJ) {
#pragma unroll 4
        for (int j = j0; j < j0 + KF_PJ; j++) {
          const uint2 b = *(const uint2*)(bp + j * 256);
#pragma unroll
          for (int c = 0; c < 4; c++) {
            uint32_t d[4];
            pf_mma(d, a[c], b);
            pf_absorb<NC>(y[c], z[c], d[0], d[1], d[2], d[3], key);
          }
        }
        q += KF_PJ;
        if (poll > 0 && (left -= KF_PJ) <= 0) {
          left = poll;
          if (__shfl_sync(~0u, fr, 0)) {
            if (q < full) {
              stop = true;
              break;
            }
          } else {
            fr = ld_flag(flag);
          }
        }
      }
    }
    const uint64_t K = dseed(c0, c1, tile);
    uint64_t h = 0;
#pragma unroll
    for (int c = 0; c < 4; c++)
      h ^= pf_fold(K, (uint32_t)((w * 32 + lane) * 4 + c), NC == 2 ? (uint64_t)z[c] << 32 | y[c] : (uint64_t)y[c]);
#pragma unroll
    for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(~0u, h, o);
    if (lane == 0) tp_dfill_k_put(rec, qrec, (rbase + i) * PF_W + w, h, q);
    mine += q;
    qlast = q;
  }
  const int part = kf_bar_or(bar, started > 0 && qlast < full);
  // the two counters as per-warp plain stores (a store-only put holds no atomic): this warp's pairs at this collective,
  // and its group's whole-tile pairs once (w == 0). The host sums over collectives and warps (pf_total_sum)
  if (lane == 0)
    tp_dfill_k_put_total(total, 2 * (((size_t)slot * ng + gid) * PF_W + w), mine,
                         w == 0 ? (unsigned long long)(started - part) * PF_W * full : 0ull);
  if (threadIdx.x % (32 * PF_W) == 0) tp_dfill_k_put_cnt(cnt, (size_t)slot * ng + gid, started);
}

template <int NC>
__global__ void __launch_bounds__(32 * PF_W * KF_G, 1)
    kfill_k(uint64_t c0, uint64_t c1, const unsigned long long* __restrict__ epoch, int slot, int nslots, int tiles, int maxcap,
            int S, int poll, const int* flag, unsigned long long* __restrict__ rec, int* __restrict__ qrec,
            int* __restrict__ cnt, unsigned long long* total) {
  tp_dfill_k<NC>(c0, c1, epoch, slot, nslots, tiles, maxcap, S, poll, flag, rec, qrec, cnt, total);
}

// ---------------------------------------------------------------- the deferred noised pass
// Decode's deferred pass: the step serves clean, stores its operands, and the noised products are computed later,
// batched over a sub-round of T tokens. This pass is those noised products: an int8 m16n8k32 IMMA (C = RZ) over the stored
// operands, with the adopted noise formed on load (the served side dithered, X = ⌊x/s⌋ + b; the other peeled with U). It is
// "receipts only" (the design's allowed simplification): each k32 output is folded unaccumulated into the same ram receipt
// pfill uses, so a chunk is 16·rows × TN × 32 and the deferred MAC count is chunks × PF_W · full · 4 · 4096; no output sum
// is kept and the peel's correction product (≈ 32/K of the MACs, int32) is not run. The operands are synthetic (fill_s8),
// one chunk's worth reused with wraparound (TPC_DWRAP chunks), so the footprint exceeds L2 and the loads are real DRAM.
//
// A chunk's B (the dithered, served side) is 1024 columns × 32, column-major in shared (Bs[col·32 + k]), formed dB + dBias;
// its A (the U side) is the warp's Sd fragments of 16 rows × 32, read from DRAM into registers in the m16n8k32 layout and
// formed dA + dU. Sd = T / (PF_W · 16): a chunk's PF_W · Sd · 16 M-rows share one B load, so a larger sub-round amortises
// the weight load over more tokens (the T effect). The fold keys are one SHA block a chunk; the receipt is dseed(c0, c1,
// chunk)'s fold, like pfill's, so tp_defer_ref_warp (host) replays it.
constexpr uint64_t TP_DEFER_DOM = 0x54502D4445464552ull;  // "TP-DEFER"
constexpr int TP_DR_MAX = 4096;                           // chunks whose warp receipts are recorded (for the sample gate)
// per-byte signed add of two s8x4 words, no saturation (the operands stay in [-127, 127], so no byte overflows)
__device__ __host__ __forceinline__ uint32_t tp_add_s8x4(uint32_t a, uint32_t b) {
  return ((a & 0x7f7f7f7fu) + (b & 0x7f7f7f7fu)) ^ ((a ^ b) & 0x80808080u);
}

// One group claims deferred chunks from *dctr (a step-wide counter, capped at dcap) and runs each like a pfill tile, except
// the operands are stored + formed noise. phase picks dtotal[phase] (0 the gap beside a collective, 1 the end-of-step drain).
// poll >= 0 polls `flag` and stops mid-chunk when it is up (the gap); poll < 0 never stops (the drain), and flag may be null.
// maxchunks caps the chunks a group runs in one launch (poll >= 0): a bounded beside-collective launch yields its SMs so the
// main stream can advance and raise the flag (an unbounded launch hogs every SM and the step serialises behind it). The drain
// (poll < 0) passes a large cap and runs the whole remainder.
template <int NC>
__device__ __noinline__ void tp_defer_run(uint64_t c0, uint64_t c1, int Sd, int poll, const int* flag,
                                          const int8_t* __restrict__ dA, const int8_t* __restrict__ dB,
                                          const int8_t* __restrict__ dU, const int8_t* __restrict__ dBias, int dwrap,
                                          int* __restrict__ dctr, int dcap, int phase, int maxchunks,
                                          unsigned long long* __restrict__ drec, int* __restrict__ dqrec,
                                          unsigned long long* dtotal) {
  extern __shared__ __align__(16) uint8_t kf_smem[];
  const int grp = threadIdx.x / (32 * PF_W), bar = 1 + grp, tg = threadIdx.x % (32 * PF_W);
  uint8_t* Bs = kf_smem + grp * KF_SMEM_GRP;
  PfKeys* ks = reinterpret_cast<PfKeys*>(Bs + PF_TN * 32 + PF_W * 32 * 16 * 4);
  const int lane = threadIdx.x & 31, w = (threadIdx.x >> 5) % PF_W, g = lane >> 2, t = lane & 3;
  const int full = Sd / 4 * KF_PAIRS_J;
  __shared__ int claim[KF_G];
  unsigned long long mine = 0;
  int fr = poll >= 0 ? ld_flag(flag) : 0;
  for (int done = 0;; done++) {
    if (kf_bar_or(bar, fr)) break;  // flag up between chunks: stop
    if (poll >= 0 && done >= maxchunks) break;  // per-launch budget: yield the SMs between gaps
    if (poll >= 0) fr = ld_flag(flag);
    if (tg == 0) claim[grp] = atomicAdd(dctr, 1);
    kf_bar(bar);
    const int idx = claim[grp];
    if (idx >= dcap) break;
    const uint64_t chunk = (uint64_t)idx;
    const int wc = idx % dwrap;
    if (tg == 0) {
      uint64_t st[8];
      pf_block(TP_DEFER_DOM, c0, c1, chunk, PF_KEY, st);
      *ks = pf_keys(st);
    }
    const uint4* sB = (const uint4*)(dB + (size_t)wc * PF_TN * 32);
    const uint4* sBi = (const uint4*)(dBias + (size_t)wc * PF_TN * 32);
    uint4* dst = (uint4*)Bs;
    for (int i = tg; i < PF_TN * 32 / 16; i += 32 * PF_W) {
      const uint4 v = sB[i], n = sBi[i];
      dst[i] = make_uint4(tp_add_s8x4(v.x, n.x), tp_add_s8x4(v.y, n.y), tp_add_s8x4(v.z, n.z), tp_add_s8x4(v.w, n.w));
    }
    kf_bar(bar);
    const PfKeys key = *ks;
    const size_t Abase = (((size_t)wc * PF_W + w) * Sd) * 512;
    uint32_t y[4], z[4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const uint32_t pos = (uint32_t)((w * 32 + lane) * 4 + c);
      y[c] = pf_y0(pos);
      z[c] = pf_z0(pos);
    }
    const uint8_t* bp = Bs + g * 32 + 8 * t;
    int q = 0, left = poll;
    bool stop = false;
#pragma unroll 1
    for (int f = 0; f < Sd && !stop; f += 4) {
      uint32_t a[4][4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const size_t Af = Abase + (size_t)(f + c) * 512;
        const int8_t* pA = dA + Af;
        const int8_t* pU = dU + Af;
        a[c][0] = tp_add_s8x4(*(const uint32_t*)(pA + g * 32 + 4 * t), *(const uint32_t*)(pU + g * 32 + 4 * t));
        a[c][1] = tp_add_s8x4(*(const uint32_t*)(pA + (g + 8) * 32 + 4 * t), *(const uint32_t*)(pU + (g + 8) * 32 + 4 * t));
        a[c][2] = tp_add_s8x4(*(const uint32_t*)(pA + g * 32 + 16 + 4 * t), *(const uint32_t*)(pU + g * 32 + 16 + 4 * t));
        a[c][3] = tp_add_s8x4(*(const uint32_t*)(pA + (g + 8) * 32 + 16 + 4 * t), *(const uint32_t*)(pU + (g + 8) * 32 + 16 + 4 * t));
      }
#pragma unroll 1
      for (int j0 = 0; j0 < KF_PAIRS_J; j0 += KF_PJ) {
#pragma unroll 4
        for (int j = j0; j < j0 + KF_PJ; j++) {
          const uint2 b = *(const uint2*)(bp + j * 256);
#pragma unroll
          for (int c = 0; c < 4; c++) {
            uint32_t d[4];
            pf_mma(d, a[c], b);
            pf_absorb<NC>(y[c], z[c], d[0], d[1], d[2], d[3], key);
          }
        }
        q += KF_PJ;
        if (poll > 0 && (left -= KF_PJ) <= 0) {
          left = poll;
          if (__shfl_sync(~0u, fr, 0)) {
            if (q < full) { stop = true; break; }
          } else {
            fr = ld_flag(flag);
          }
        }
      }
    }
    const uint64_t K = dseed(c0, c1, chunk);
    uint64_t h = 0;
#pragma unroll
    for (int c = 0; c < 4; c++)
      h ^= pf_fold(K, (uint32_t)((w * 32 + lane) * 4 + c), NC == 2 ? (uint64_t)z[c] << 32 | y[c] : (uint64_t)y[c]);
#pragma unroll
    for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(~0u, h, o);
    if (lane == 0 && idx < TP_DR_MAX) {
      drec[(size_t)idx * PF_W + w] = h;
      dqrec[(size_t)idx * PF_W + w] = q;
    }
    mine += q;
  }
  if (lane == 0) atomicAdd(dtotal + phase, mine);
}

// beside a collective: the deferred pass alone (its step quota, bounded by dmaxc so it yields the SMs, stopping at the
// flag). The filler is launched behind it (dpass_launch), not tail-called here, so no deferred MMA or store is on the
// filler warps' path and the scorer's `filler_paths` gate passes (SCORER v5, the put split). Both read the same flag.
template <int NC>
__global__ void __launch_bounds__(32 * PF_W * KF_G, 1)
    tp_dpass_k(uint64_t c0, uint64_t c1, int Sd, int dmaxc, const int* flag, const int8_t* dA, const int8_t* dB,
               const int8_t* dU, const int8_t* dBias, int dwrap, int* dctr, int dcap, unsigned long long* drec, int* dqrec,
               unsigned long long* dtotal) {
  // the deferred pass polls the flag between chunks only (poll 0): a chunk computes a real noised product, so it is atomic —
  // none is lost to a mid-chunk cut, and the leftover goes to the drain. dmaxc bounds this launch so it yields the SMs; the
  // filler (kfill_k, launched next) then cuts mid-tile as before.
  tp_defer_run<NC>(c0, c1, Sd, 0, flag, dA, dB, dU, dBias, dwrap, dctr, dcap, 0, dmaxc, drec, dqrec, dtotal);
}

// the drain: the deferred quota on every SM with no stop and no filler (the end-of-step overflow in the defer arm, or the
// whole sub-round's share in the dalone arm). phase 1 (dtotal[1]).
template <int NC>
__global__ void __launch_bounds__(32 * PF_W * KF_G, 1)
    tp_drain_k(uint64_t c0, uint64_t c1, int Sd, const int8_t* dA, const int8_t* dB, const int8_t* dU, const int8_t* dBias,
               int dwrap, int* dctr, int dcap, unsigned long long* drec, int* dqrec, unsigned long long* dtotal) {
  tp_defer_run<NC>(c0, c1, Sd, -1, nullptr, dA, dB, dU, dBias, dwrap, dctr, dcap, 1, dcap, drec, dqrec, dtotal);
}

// a step's deferred counter back to 0 (the step's fresh demand); dtotal accumulates across the timed steps and is zeroed
// once before timing, so defer_gap/defer_drain average over (reps + 2) steps
__global__ void dprolog_k(int* dctr) {
  if (threadIdx.x == 0) *dctr = 0;
}

// warp w's receipt of tile `tile` after q pairs (pf_ref's chains, cut at q): the host's side of a record
static uint64_t kf_ref_warp(uint64_t c0, uint64_t c1, uint64_t tile, int S, int w, int q) {
  pf_host_k();
  uint64_t st[8];
  std::vector<uint64_t> Bb(PF_TN, 0);
  for (int b = 0; b < PF_TN / 16; b++) {
    pf_block_h(PF_DOM01, c0, c1, tile, PF_B + (uint64_t)b, st);
    for (int i = 0; i < 16; i++) Bb[16 * b + i] = pf_row(st, i);
  }
  pf_block_h(PF_DOM01, c0, c1, tile, PF_KEY, st);
  const PfKeys key = pf_keys(st);
  const uint64_t K = dseed(c0, c1, tile);
  std::vector<uint64_t> Ar(16 * (size_t)S);
  for (int f = 0; f < S; f++) {
    pf_block_h(PF_DOM01, c0, c1, tile, (uint64_t)w * S + f, st);
    for (int i = 0; i < 16; i++) Ar[16 * f + i] = pf_row(st, i);
  }
  uint64_t h = 0;
  for (int lane = 0; lane < 32; lane++) {
    const int g = lane >> 2, t = lane & 3;
    for (int c = 0; c < 4; c++) {
      const uint32_t pos = (uint32_t)((w * 32 + lane) * 4 + c);
      uint32_t y = pf_y0(pos), z = pf_z0(pos);
      for (int f = c; f < S; f += 4) {
        const int jn = std::min(KF_PAIRS_J, std::max(0, q - (f / 4) * KF_PAIRS_J));
        const uint64_t R0 = Ar[16 * f + g], R1 = Ar[16 * f + g + 8];
        for (int j = 0; j < jn; j++) {
          const uint64_t *b0 = &Bb[8 * j + 2 * t], *b1 = b0 + 1;  // the bits tile's one plane a column
          pf_absorb<TP_NC>(y, z, pf_word<PF_BITS>(R0, b0), pf_word<PF_BITS>(R0, b1), pf_word<PF_BITS>(R1, b0),
                       pf_word<PF_BITS>(R1, b1), key);
        }
      }
      h ^= pf_fold(K, pos, TP_NC == 2 ? ((uint64_t)z << 32 | y) : (uint64_t)y);
    }
  }
  return h;
}

// the deferred pass's synthetic operands, the seeds fill_s8 fills them with (host and device must agree)
constexpr uint64_t TP_SA = 0xDEFA0001ull, TP_SB = 0xDEFB0002ull, TP_SU = 0xDEFC0003ull, TP_SBIAS = 0xDEFD0004ull;
// fill_s8's byte i on the host (the device kernel's p[i])
static inline int8_t h_fill_s8(uint64_t seed, size_t i) {
  uint64_t x = (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull ^ seed;
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
  return (int8_t)((int)(x & 0xff) - 128);
}
static inline uint32_t h_word(uint64_t seed, size_t a) {  // the 4-byte word at byte offset a (aligned)
  uint32_t w = 0;
  for (int b = 0; b < 4; b++) w |= ((uint32_t)(uint8_t)h_fill_s8(seed, a + b)) << (8 * b);
  return w;
}
// the deferred operand's formed s8 byte at linear offset o: tp_add_s8x4(stored word, noise word), byte o % 4
static inline int h_formed(uint64_t sStore, uint64_t sNoise, size_t o) {
  const size_t a = o & ~(size_t)3;
  const uint32_t fw = tp_add_s8x4(h_word(sStore, a), h_word(sNoise, a));
  return (int8_t)((fw >> (8 * (o & 3))) & 0xff);
}
// warp w's deferred-chunk receipt (tp_defer_run's math on the host), fragments f < fraglimit, pairs cut at q
static uint64_t tp_defer_ref_warp(uint64_t c0, uint64_t c1, uint64_t chunk, int wc, int w, int Sd, int fraglimit, int q) {
  pf_host_k();
  uint64_t st[8];
  pf_block_h(TP_DEFER_DOM, c0, c1, chunk, PF_KEY, st);
  const PfKeys key = pf_keys(st);
  const uint64_t K = dseed(c0, c1, chunk);
  const size_t Bbase = (size_t)wc * PF_TN * 32;
  const size_t Abase = (((size_t)wc * PF_W + w) * Sd) * 512;
  std::vector<int> B((size_t)PF_TN * 32);  // formedB[col * 32 + k]
  for (int col = 0; col < PF_TN; col++)
    for (int k = 0; k < 32; k++) B[(size_t)col * 32 + k] = h_formed(TP_SB, TP_SBIAS, Bbase + (size_t)col * 32 + k);
  auto Aval = [&](int F, int row, int k) { return h_formed(TP_SA, TP_SU, Abase + (size_t)F * 512 + (size_t)row * 32 + k); };
  // B's shared layout stores a column's s8 words in the order 0 4 1 5 2 6 3 7 (pf_tiles), so the 8-byte fragment load gives
  // the IMMA its canonical B fragment; the deferred pass copies dB into shared byte for byte, so logical k reads physical
  // offset bphys(k) (A is stored and read in canonical row order, so Aval stays by k).
  auto bphys = [](int k) { const int m = k >> 2, r = k & 3; return ((m < 4 ? 2 * m : 2 * (m - 4) + 1) << 2) + r; };
  uint64_t h = 0;
  for (int lane = 0; lane < 32; lane++) {
    const int g = lane >> 2, t = lane & 3;
    for (int c = 0; c < 4; c++) {
      const uint32_t pos = (uint32_t)((w * 32 + lane) * 4 + c);
      uint32_t y = pf_y0(pos), z = pf_z0(pos);
      for (int F = c; F < fraglimit; F += 4) {
        const int jn = std::min(KF_PAIRS_J, std::max(0, q - (F / 4) * KF_PAIRS_J));
        for (int j = 0; j < jn; j++) {
          int d[4] = {0, 0, 0, 0};
          const int c0col = 8 * j + 2 * t, c1col = 8 * j + 2 * t + 1;
          for (int k = 0; k < 32; k++) {
            const int rg = Aval(F, g, k), rg8 = Aval(F, g + 8, k);
            const int kb = bphys(k);
            const int b0 = B[(size_t)c0col * 32 + kb], b1 = B[(size_t)c1col * 32 + kb];
            d[0] += rg * b0; d[1] += rg * b1; d[2] += rg8 * b0; d[3] += rg8 * b1;
          }
          pf_absorb<TP_NC>(y, z, (uint32_t)d[0], (uint32_t)d[1], (uint32_t)d[2], (uint32_t)d[3], key);
        }
      }
      h ^= pf_fold(K, pos, TP_NC == 2 ? ((uint64_t)z << 32 | y) : (uint64_t)y);
    }
  }
  return h;
}

// ---------------------------------------------------------------- one rank's state
struct PFill {  // the phase filler
  bool on = false;
  int NC = 4, S = 64, tiles = 2, tiles_g = 16, poll = KF_PJ, grid = 0;
  int nslots = 0;
  int* flags = nullptr;
  unsigned long long* epoch = nullptr;
  unsigned long long* total = nullptr;
  unsigned long long* rec = nullptr;  // nslots x groups x maxcap x PF_W: warp receipts
  int* qrec = nullptr;                // the same: pairs each warp ran
  int* cnt = nullptr;                 // nslots x groups: tiles started
  cudaStream_t side = nullptr;
  cudaEvent_t fork = nullptr, join = nullptr;
  long long groups() const { return (long long)grid * KF_G; }
  int full() const { return S / 4 * KF_PAIRS_J; }
  int cap(int slot) const { return slot == nslots - 1 ? tiles_g : tiles; }
  int maxcap() const { return std::max(tiles, tiles_g); }
  long long stride() const { return groups() * maxcap(); }  // tiles a collective
  unsigned long long tile(unsigned long long ep, int slot, long long gid, int i) const {
    return (ep * nslots + slot) * (unsigned long long)stride() + (unsigned long long)gid * maxcap() + i;
  }
  size_t rix(int slot, long long gid, int i, int w) const {
    return (((size_t)slot * groups() + gid) * maxcap() + i) * PF_W + w;
  }
};

struct DPass {  // the deferred noised pass
  bool on = false;
  int Sd = 8;        // fragments a warp a chunk; Sd = T / (PF_W * 16), so a chunk is PF_W * Sd * 16 token-rows
  int T = 1024;      // the sub-round's tokens
  int dwrap = 512;   // distinct chunks of synthetic operands in DRAM (the footprint exceeds L2)
  int grid = 0;      // the combined kernel's grid beside a collective (= the filler's, sms - reserve)
  int dgrid = 0;     // the drain's grid (every SM)
  int dmaxc = 4;     // chunks a group a beside-collective launch (bounds it so it yields the SMs between gaps)
  int dcap = 0;      // a step's chunk quota (set per point)
  int* dctr = nullptr;
  unsigned long long* dtotal = nullptr;  // [0] the pairs run in the gaps, [1] the pairs drained
  unsigned long long* drec = nullptr;    // TP_DR_MAX x PF_W: the recorded chunks' warp receipts
  int* dqrec = nullptr;
  int8_t *dA = nullptr, *dB = nullptr, *dU = nullptr, *dBias = nullptr;
  int fullp() const { return Sd / 4 * KF_PAIRS_J; }                     // pairs a warp a whole chunk
  long long chunk_pairs() const { return (long long)PF_W * fullp(); }   // pairs a whole chunk (all PF_W warps)
};

struct Rank {
  int r = 0, tp = 1;
  Dev d;  // dcap's: st, nv and sms are what time_work reads
  ncclComm_t comm;
  int8_t* W[T_LAYERS][4];
  int8_t* Wlm;
  int8_t* X;
  int* D;
  uint16_t* ar;       // the BF16 hidden state the all-reduces sum
  uint16_t* logits;   // B x VOCAB/TP BF16
  uint16_t* gath;     // B x VOCAB BF16
  int8_t* kv;
  size_t kv_bytes;
  int8_t *q, *P;
  int *S, *O;
  int LN[4], LK[4];
  PFill pf;
  DPass dp;
};

static int8_t* kv_slot(const Rank& R, int Be, int L, int layer, bool v, bool* wrapped) {
  const size_t per = (size_t)Be * HKV * L * HD, lay = 2 * per;
  const size_t nfit = R.kv_bytes / lay;
  if (nfit < (size_t)T_LAYERS) *wrapped = true;
  const size_t slot = nfit ? (size_t)layer % nfit : 0;
  return R.kv + slot * lay + (v ? per : 0);
}

static void glue(Rank& R, const void* in, size_t in_bytes, void* out, size_t out_bytes, int* flag, cudaStream_t st) {
  const size_t nin = in_bytes / 16, nout = out_bytes / 16, n = std::max(nin, nout);
  const unsigned blocks = (unsigned)std::max<size_t>(1, std::min<size_t>(R.d.sms, (n + 255) / 256));
  glue_k<<<blocks, 256, 0, st>>>((const int4*)in, nin, (int4*)out, nout, flag);
}

static void kfill_launch(const PFill& f, int slot, uint64_t c0, uint64_t c1, cudaStream_t st) {
  kfill_k<TP_NC><<<f.grid, 32 * PF_W * KF_G, KF_SMEM, st>>>(c0, c1, f.epoch, slot, f.nslots, f.cap(slot), f.maxcap(), f.S, f.poll,
                                                        f.flags + slot, f.rec, f.qrec, f.cnt, f.total);
}

// the filler's per-warp counter buffer (f.total): 2 u64 a warp of a collective, gw = (slot·groups + gid)·PF_W + w. A
// store-only put can't atomic-accumulate, so each collective's launch overwrites its own slots and the host sums them
// (over collectives and warps: the last step's work, one launch's when nslots is 1)
static size_t pf_total_slots(const PFill& f) { return 2 * (size_t)f.nslots * f.groups() * PF_W; }
static size_t pf_total_bytes(const PFill& f) { return pf_total_slots(f) * sizeof(unsigned long long); }
static void pf_total_sum(const PFill& f, unsigned long long tot[2]) {
  const size_t n = pf_total_slots(f);
  std::vector<unsigned long long> buf(n);
  CK(cudaMemcpy(buf.data(), f.total, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  tot[0] = tot[1] = 0;
  for (size_t i = 0; i < n; i += 2) { tot[0] += buf[i]; tot[1] += buf[i + 1]; }
}

static uint64_t g_tc0 = 0x0123456789abcdefull, g_tc1 = 0xfedcba9876543210ull;  // the round's coins (DFILL_COINS)

// beside a collective: the deferred pass (its step quota, R.dp.dcap, bounded by dmaxc, stopping at the flag), then the
// filler (kfill_k) fills the gap it left — two launches on `st`, so the filler warps' path holds no deferred MMA or store
// (SCORER v5 `filler_paths`); both read f.flags + slot
static void dpass_launch(Rank& R, int slot, cudaStream_t st) {
  const DPass& dp = R.dp;
  const PFill& f = R.pf;
  tp_dpass_k<TP_NC><<<dp.grid, 32 * PF_W * KF_G, KF_SMEM, st>>>(
      g_tc0, g_tc1, dp.Sd, dp.dmaxc, f.flags + slot, dp.dA, dp.dB, dp.dU, dp.dBias, dp.dwrap, dp.dctr, dp.dcap,
      dp.drec, dp.dqrec, dp.dtotal);
  kfill_launch(f, slot, g_tc0, g_tc1, st);
}

// a collective at slot `slot`: side work forks beside it (1 the filler alone, 2 the deferred pass then the filler); `fn`
// enqueues the NCCL call on st
static void collective(Rank& R, int sidework, int slot, cudaStream_t st, const std::function<void()>& fn) {
  if (sidework) {
    CK(cudaEventRecord(R.pf.fork, st));
    CK(cudaStreamWaitEvent(R.pf.side, R.pf.fork, 0));
    if (sidework == 2) dpass_launch(R, slot, R.pf.side);
    else kfill_launch(R.pf, slot, g_tc0, g_tc1, R.pf.side);
  }
  fn();
}

// kind: 0 the step, 1 without collectives, 2 the collectives alone; sidework: 0 none, 1 the filler beside each collective,
// 2 the deferred pass then the filler, with the step's deferred overflow drained after the last collective
static void enqueue_step(Rank& R, const std::vector<Lt>& p, const Lt& lm, int kind, int sidework, int B, int L, bool* wrapped) {
  cudaStream_t st = R.d.st;
  const bool pf = sidework != 0;
  const size_t hb = (size_t)B * T_HID * 2;  // the BF16 hidden state
  const int Be = B / R.tp;
  const bool comp = kind != 2, comm = kind != 1;
  int slot = 0;
  int* flag = nullptr;  // the flag the next small kernel raises
  auto coll_ar = [&] {
    if (!comm) return;
    collective(R, sidework, slot, st, [&] { NK(ncclAllReduce(R.ar, R.ar, (size_t)B * T_HID, ncclBfloat16, ncclSum, R.comm, st)); });
    flag = pf ? R.pf.flags + slot : nullptr;
    slot++;
  };
  auto g = [&](const void* in, size_t ib, void* out, size_t ob) {
    if (!comp) {
      if (flag) glue(R, R.ar, 16, R.D, 16, flag, st);  // the collectives alone still release their fillers
      flag = nullptr;
      return;
    }
    glue(R, in, ib, out, ob, flag, st);
    flag = nullptr;
  };
  if (pf) prolog_k<<<1, 256, 0, st>>>(R.pf.flags, R.pf.nslots, R.pf.epoch);
  if (sidework == 2) dprolog_k<<<1, 32, 0, st>>>(R.dp.dctr);
  coll_ar();  // the vocab-parallel embedding
  for (int l = 0; l < T_LAYERS; l++) {
    g(R.ar, hb, R.X, (size_t)B * T_HID);  // residual + RMSNorm + quantize
    if (comp) lt_run(p[0], R.X, R.W[l][0], R.D, st);
    g(R.D, (size_t)B * R.LN[0] * 4, R.q, (size_t)B * R.LN[0]);  // RoPE + cache write
    if (comp) {
      const int8_t* K = kv_slot(R, Be, L, l, false, wrapped);
      const int8_t* V = kv_slot(R, Be, L, l, true, wrapped);
      qk_dec<<<dim3((L + QK_W * QK_KW - 1) / (QK_W * QK_KW), HKV, Be), 256, 0, st>>>(K, R.q, R.S, L);
      CK(cudaMemsetAsync(R.O, 0, (size_t)Be * HQ * HD * sizeof(int), st));
      pv_dec<<<dim3((L / 4 + PV_W * PV_GW - 1) / (PV_W * PV_GW), HKV, Be), 256, 0, st>>>(V, R.P, R.O, L);
    }
    g(R.O, (size_t)B * R.LK[1] * 4, R.X, (size_t)B * R.LK[1]);  // quantize the attention output
    if (comp) lt_run(p[1], R.X, R.W[l][1], R.D, st);
    g(R.D, (size_t)B * T_HID * 4, R.ar, hb);  // the o shard's partial sum, BF16
    coll_ar();
    g(R.ar, 2 * hb, R.X, (size_t)B * T_HID);  // residual + RMSNorm + quantize
    if (comp) lt_run(p[2], R.X, R.W[l][2], R.D, st);
    g(R.D, (size_t)B * R.LN[2] * 4, R.X, (size_t)B * R.LK[3]);  // SiLU-and-mul + quantize
    if (comp) lt_run(p[3], R.X, R.W[l][3], R.D, st);
    g(R.D, (size_t)B * T_HID * 4, R.ar, hb);  // the down shard's partial sum, BF16
    coll_ar();
  }
  g(R.ar, hb, R.X, (size_t)B * T_HID);  // final norm + quantize
  if (comp) lt_run(lm, R.X, R.Wlm, R.D, st);
  g(R.D, (size_t)B * (T_VOCAB / R.tp) * 4, R.logits, (size_t)B * (T_VOCAB / R.tp) * 2);
  if (comm) {
    collective(R, pf, slot, st, [&] {
      NK(ncclAllGather(R.logits, R.gath, (size_t)B * (T_VOCAB / R.tp), ncclBfloat16, R.comm, st));
    });
    flag = pf ? R.pf.flags + slot : nullptr;
    slot++;
    glue(R, R.gath, 16, R.D, 16, flag, st);  // the sampler's first kernel
  }
  if (pf) {
    CK(cudaEventRecord(R.pf.join, R.pf.side));
    CK(cudaStreamWaitEvent(st, R.pf.join, 0));
  }
  // the defer arm drains the step's deferred overflow (the quota the gaps did not reach) on every SM, no filler: its time
  // is part of the step, so the slowdown carries the overflow's cost
  if (sidework == 2)
    tp_drain_k<TP_NC><<<R.dp.dgrid, 32 * PF_W * KF_G, KF_SMEM, st>>>(g_tc0, g_tc1, R.dp.Sd, R.dp.dA, R.dp.dB, R.dp.dU, R.dp.dBias,
                                                                R.dp.dwrap, R.dp.dctr, R.dp.dcap, R.dp.drec, R.dp.dqrec,
                                                                R.dp.dtotal);
}

// the dalone arm: a step's whole deferred quota drained on every SM with no decode and no gaps — the deferred pass's
// decode cost without the gaps (dtotal[1])
static void enqueue_dalone(Rank& R) {
  cudaStream_t st = R.d.st;
  dprolog_k<<<1, 32, 0, st>>>(R.dp.dctr);
  tp_drain_k<TP_NC><<<R.dp.dgrid, 32 * PF_W * KF_G, KF_SMEM, st>>>(g_tc0, g_tc1, R.dp.Sd, R.dp.dA, R.dp.dB, R.dp.dU, R.dp.dBias,
                                                              R.dp.dwrap, R.dp.dctr, R.dp.dcap, R.dp.drec, R.dp.dqrec,
                                                              R.dp.dtotal);
}

static double now_s() { return now_ms() * 1e-3; }

// every rank meets here (a one-element all-reduce, then the stream drained)
static void barrier(Rank& R) {
  NK(ncclAllReduce(R.ar, R.ar, 1, ncclBfloat16, ncclSum, R.comm, R.d.st));
  CK(cudaStreamSynchronize(R.d.st));
}

static int getenv_i(const char* k, int dflt) { return getenv(k) ? atoi(getenv(k)) : dflt; }

// raises *flag once `cycles` SM clocks have passed: the cut gate's stand-in for the kernel after a collective
__global__ void raise_after_k(int* flag, long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
  *(volatile int*)flag = 1;
}

// the records of one launch at `slot` (the last rep's, after an arm)
struct KRecs {
  std::vector<unsigned long long> rec;
  std::vector<int> q, cnt;
};
static KRecs kf_fetch(const PFill& f, int slot0, int nsl) {
  KRecs k;
  const size_t nr = (size_t)nsl * f.groups() * f.maxcap() * PF_W;
  k.rec.resize(nr);
  k.q.resize(nr);
  k.cnt.resize((size_t)nsl * f.groups());
  CK(cudaMemcpy(k.rec.data(), f.rec + f.rix(slot0, 0, 0, 0), nr * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(k.q.data(), f.qrec + f.rix(slot0, 0, 0, 0), nr * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(k.cnt.data(), f.cnt + (size_t)slot0 * f.groups(), k.cnt.size() * 4, cudaMemcpyDeviceToHost));
  return k;
}
// what a launch's records must look like whatever the flag did: tiles started within the cap, every tile but the last
// whole, the last's warps at q in [KF_PJ, full] on the poll's grid; *partial counts the warps cut mid-tile
static long kf_shape(const PFill& f, const KRecs& k, int slot0, int nsl, long* partial, unsigned long long* pairs,
                     unsigned long long* whole) {
  long bad = 0;
  *partial = 0;
  *pairs = *whole = 0;
  const int full = f.full();
  for (int s = 0; s < nsl; s++)
    for (long long gi = 0; gi < f.groups(); gi++) {
      const int c = k.cnt[(size_t)s * f.groups() + gi];
      bad += c < 0 || c > f.cap(slot0 + s);
      bool cut = false;
      for (int i = 0; i < std::min(std::max(c, 0), f.cap(slot0 + s)); i++)
        for (int w = 0; w < PF_W; w++) {
          const int q = k.q[f.rix(s, gi, i, w)];
          *pairs += q;
          if (i < c - 1) bad += q != full;
          else {
            bad += q < KF_PJ || q > full || (f.poll > 0 && q % KF_PJ);
            *partial += q < full;
            cut |= q < full;
          }
        }
      if (c > 0) *whole += (unsigned long long)(c - cut) * PF_W * full;
    }
  return bad;
}
static unsigned long long kf_xor(const PFill& f, const KRecs& k, int s, long long gi, int i) {
  unsigned long long x = 0;
  for (int w = 0; w < PF_W; w++) x ^= k.rec[f.rix(s, gi, i, w)];
  return x;
}
struct KCheck {
  long whole = 0, whole_bad = 0, whole_ctl = 0, warps = 0, warps_bad = 0, warps_ctl = 0;
  bool clean() const { return !whole_bad && !whole_ctl && !warps_bad && !warps_ctl; }
};
// up to n whole tiles (their warps' receipts XOR to pfill's dfill_ref; dfill_ref at S - 8 must differ) and n cut warps
// (kf_ref_warp at their q; at q - KF_PJ must differ) of the launch(es) at slot0.., drawn with rng
static KCheck kf_sample(const PFill& f, const KRecs& k, int slot0, int nsl, unsigned long long ep, int n, std::mt19937_64& rng) {
  KCheck r;
  const int full = f.full();
  for (int it = 0; it < 4000 * n && (r.whole < n || r.warps < n); it++) {
    const int s = (int)(rng() % nsl);
    const long long gi = (long long)(rng() % f.groups());
    const int c = std::min(k.cnt[(size_t)s * f.groups() + gi], f.cap(slot0 + s));
    if (c <= 0) continue;
    const int i = (int)(rng() % c);
    const unsigned long long tile = f.tile(ep, slot0 + s, gi, i);
    bool whole = true;
    for (int w = 0; w < PF_W; w++) whole &= k.q[f.rix(s, gi, i, w)] == full;
    if (whole && r.whole < n) {
      const unsigned long long x = kf_xor(f, k, s, gi, i);
      r.whole_bad += x != dfill_ref(g_tc0, g_tc1, tile, TP_NC, f.S);
      r.whole_ctl += x == dfill_ref(g_tc0, g_tc1, tile, TP_NC, f.S - 8);
      r.whole++;
    } else if (!whole && r.warps < n) {
      for (int w = 0; w < PF_W; w++) {
        const int q = k.q[f.rix(s, gi, i, w)];
        if (q == full) continue;
        const unsigned long long v = k.rec[f.rix(s, gi, i, w)];
        r.warps_bad += v != kf_ref_warp(g_tc0, g_tc1, tile, f.S, w, q);
        r.warps_ctl += v == kf_ref_warp(g_tc0, g_tc1, tile, f.S, w, q - KF_PJ);
        r.warps++;
        break;
      }
    }
  }
  return r;
}

// The phase filler's gate on this rank, before anything is timed, at slot 3 with TPC_TILES (2 unless set) a group:
//   fixed  poll -1: every group runs its tiles whole; every 97th tile's warp receipts XOR to pfill's dfill_ref (and not to
//          its S - 8 control), and one tile's warps each equal kf_ref_warp at q = full;
//   up     the flag already up: no group starts a tile, no record is written (0xA5 stays), the totals stay 0;
//   cut    the flag raised TPC_CUT_CYC SM clocks after the launch (a third of a tile at S 64): every group's records have
//          the shape kf_shape asks, at least one warp is cut, the totals equal the records' pairs, and 64 cut warps and
//          whole tiles are checked (each against its q - KF_PJ control).
static int pf_gate(Rank& R) {
  PFill& f = R.pf;
  int fails = 0;
  const int keep_poll = f.poll, slot = 3, full = f.full();
  const size_t nr = (size_t)f.groups() * f.maxcap() * PF_W;
  unsigned long long ep;
  CK(cudaMemcpy(&ep, f.epoch, sizeof ep, cudaMemcpyDeviceToHost));
  std::mt19937_64 rng(0x5EED0000ull + R.r);
  raise_after_k<<<1, 32, 0, R.d.st>>>(f.flags + slot, 0);
  CK(cudaDeviceSynchronize());
  for (const char* kind : {"fixed", "up", "cut"}) {
    const std::string kd = kind;
    f.poll = kd == "fixed" ? -1 : KF_PJ;
    CK(cudaMemset(f.rec + f.rix(slot, 0, 0, 0), 0xA5, nr * 8));
    CK(cudaMemset(f.qrec + f.rix(slot, 0, 0, 0), 0xA5, nr * 4));
    CK(cudaMemset(f.cnt + (size_t)slot * f.groups(), 0xA5, f.groups() * 4));
    CK(cudaMemset(f.total, 0, pf_total_bytes(f)));
    CK(cudaMemset(f.flags, 0, f.nslots * sizeof(int)));
    if (kd == "up") CK(cudaMemset(f.flags + slot, 1, 1));
    CK(cudaDeviceSynchronize());
    kfill_launch(f, slot, g_tc0, g_tc1, R.pf.side);
    if (kd == "cut") raise_after_k<<<1, 32, 0, R.d.st>>>(f.flags + slot, getenv_i("TPC_CUT_CYC", 200000));
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    const KRecs k = kf_fetch(f, slot, 1);
    unsigned long long tot[2];
    pf_total_sum(f, tot);
    long bad = 0, partial = 0, cnt_bad = 0, untouched_bad = 0;
    KCheck ck;
    unsigned long long pairs = 0, whole = 0;
    if (kd == "up") {
      for (int c : k.cnt) cnt_bad += c != 0;
      for (unsigned long long v : k.rec) untouched_bad += v != 0xA5A5A5A5A5A5A5A5ull;
      bad = cnt_bad + untouched_bad + (tot[0] != 0) + (tot[1] != 0);
    } else {
      bad = kf_shape(f, k, slot, 1, &partial, &pairs, &whole);
      bad += tot[0] != pairs || tot[1] != whole;
      if (kd == "fixed") {
        for (long long gi = 0; gi < f.groups(); gi++) {
          cnt_bad += k.cnt[gi] != f.tiles;
          for (int i = 0; i < f.tiles; i++) {
            if ((gi * f.tiles + i) % 97) continue;
            const unsigned long long x = kf_xor(f, k, 0, gi, i), tile = f.tile(ep, slot, gi, i);
            ck.whole_bad += x != dfill_ref(g_tc0, g_tc1, tile, TP_NC, f.S);
            ck.whole_ctl += x == dfill_ref(g_tc0, g_tc1, tile, TP_NC, f.S - 8);
            ck.whole++;
          }
        }
        for (int w = 0; w < PF_W; w++) {
          ck.warps_bad += k.rec[f.rix(0, 0, 0, w)] != kf_ref_warp(g_tc0, g_tc1, f.tile(ep, slot, 0, 0), f.S, w, full);
          ck.warps++;
        }
        bad += cnt_bad + (ck.whole == 0) + (partial != 0);
      } else {
        ck = kf_sample(f, k, slot, 1, ep, 64, rng);
        bad += partial == 0 || ck.warps == 0;
      }
    }
    const bool ok = !bad && ck.clean();
    printf("{\"mode\": \"pf_gate\", \"kind\": \"%s\", \"rank\": %d, \"poll\": %d, \"S\": %d, \"tiles\": %d, \"grid\": %d, "
           "\"groups\": %lld, \"whole_checked\": %ld, \"whole_bad\": %ld, \"whole_control_equal\": %ld, \"warps_checked\": %ld, "
           "\"warps_bad\": %ld, \"warps_control_equal\": %ld, \"warps_cut\": %ld, \"cnt_bad\": %ld, \"untouched_bad\": %ld, "
           "\"pairs\": %llu, \"pairs_whole\": %llu, \"shape_bad\": %ld, \"ok\": %s}\n",
           kind, R.r, f.poll, f.S, f.tiles, f.grid, f.groups(), ck.whole, ck.whole_bad, ck.whole_ctl, ck.warps, ck.warps_bad,
           ck.warps_ctl, partial, cnt_bad, untouched_bad, tot[0], tot[1], bad, ok ? "true" : "false");
    fflush(stdout);
    fails += !ok;
  }
  f.poll = keep_poll;
  CK(cudaMemset(f.flags, 0, f.nslots * sizeof(int)));
  return fails;
}

// after an arm, on the last rep's records: every collective's shape (kf_shape) and n whole tiles and n cut warps drawn
// over the collectives
static void pf_check(Rank& R, int n, long* shape_bad, long* cut_warps, KCheck* ck, unsigned long long* pairs_last,
                     unsigned long long* whole_last) {
  PFill& f = R.pf;
  unsigned long long ep;
  CK(cudaMemcpy(&ep, f.epoch, sizeof ep, cudaMemcpyDeviceToHost));
  const KRecs k = kf_fetch(f, 0, f.nslots);
  *shape_bad = kf_shape(f, k, 0, f.nslots, cut_warps, pairs_last, whole_last);
  std::mt19937_64 rng(ep * 7919 + R.r);
  *ck = kf_sample(f, k, 0, f.nslots, ep, n, rng);
}

// n recorded deferred chunks (q = full, drawn with rng) of the last timed step: each warp receipt against tp_defer_ref_warp,
// and a control at Sd - 4 fragments that must differ
struct DCheck {
  long whole = 0, whole_bad = 0, whole_ctl = 0;
  bool clean() const { return whole > 0 && !whole_bad && !whole_ctl; }
};
static DCheck dp_sample(Rank& R, int n) {
  DPass& dp = R.dp;
  const int nr = std::min(std::max(dp.dcap, 0), TP_DR_MAX);
  DCheck r;
  if (nr == 0) return r;
  std::vector<unsigned long long> rec((size_t)nr * PF_W);
  std::vector<int> q((size_t)nr * PF_W);
  CK(cudaMemcpy(rec.data(), dp.drec, rec.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(q.data(), dp.dqrec, q.size() * 4, cudaMemcpyDeviceToHost));
  const int full = dp.fullp();
  std::mt19937_64 rng(0xD36ull + R.r);
  for (int it = 0; it < 6000 * n && r.whole < n; it++) {
    const int idx = (int)(rng() % nr), w = (int)(rng() % PF_W);
    if (q[(size_t)idx * PF_W + w] != full) continue;
    const int wc = idx % dp.dwrap;
    const unsigned long long x = rec[(size_t)idx * PF_W + w];
    r.whole_bad += x != tp_defer_ref_warp(g_tc0, g_tc1, (uint64_t)idx, wc, w, dp.Sd, dp.Sd, full);
    r.whole_ctl += x == tp_defer_ref_warp(g_tc0, g_tc1, (uint64_t)idx, wc, w, dp.Sd, dp.Sd - 4, full);
    r.whole++;
  }
  return r;
}

static void emit(Rank& R, const char* arm, int B, int L, const std::vector<Rep>& v, double served_macs, double fill_mmas_rep,
                 const char* extra) {
  std::vector<Rep> s = v;
  std::sort(s.begin(), s.end(), [](const Rep& a, const Rep& b) { return a.ms < b.ms; });
  const Rep& m = s[s.size() / 2];
  const double peak = 1024.0 * R.d.sms * m.mhz * 1e6;
  const double fill_macs = fill_mmas_rep * 4096.0;
  printf("{\"mode\": \"tp\", \"tp\": %d, \"rank\": %d, \"B\": %d, \"L\": %d, \"arm\": \"%s\", \"ms\": %.4f, \"ms_min\": %.4f, "
         "\"ms_max\": %.4f, \"mhz\": %.1f, \"watt\": %.1f, \"served_macs\": %.6e, \"served_share\": %.5f, \"fill_mmas_rep\": %.1f, "
         "\"fill_macs_rep\": %.6e, \"fill_share\": %.5f%s, \"reps\": [",
         R.tp, R.r, B, L, arm, m.ms, s.front().ms, s.back().ms, m.mhz, m.watt, served_macs,
         served_macs / (m.ms * 1e-3) / peak, fill_mmas_rep, fill_macs, fill_macs / (m.ms * 1e-3) / peak, extra);
  for (size_t i = 0; i < v.size(); i++) printf("%s[%.4f, %.0f, %.0f]", i ? ", " : "", v[i].ms, v[i].mhz, v[i].watt);
  printf("]}\n");
  fflush(stdout);
}

// one (B, L) point: plans, then arms in the order TPC_ARMS names (default base phase nocomm comm phase base; spaces or commas)
static int run_point(Rank& R, int B, int L, int reps) {
  if (B % R.tp) { fprintf(stderr, "B %d not a multiple of TP %d\n", B, R.tp); return 1; }
  std::vector<Lt> p(4);
  for (int j = 0; j < 4; j++) p[j] = lt_make(B, R.LN[j], R.LK[j], R.X, R.W[0][j], R.D, R.d.st);
  const Lt lm = lt_make(B, T_VOCAB / R.tp, T_HID, R.X, R.Wlm, R.D, R.d.st);
  for (int j = 0; j < 4; j++)
    if (p[j].algo_id == -2) return 6;
  if (lm.algo_id == -2) return 6;
  double lin = 0;
  for (int j = 0; j < 4; j++) lin += (double)T_LAYERS * B * R.LN[j] * R.LK[j];
  lin += (double)B * (T_VOCAB / R.tp) * T_HID;
  const double att = 2.0 * T_LAYERS * B * (T_HQ / R.tp) * T_HD * (double)L;
  const double served = lin + att;
  // per-shard storage the deferred pass holds: the served-side operand of each deferred product kept over the
  // sub-round (the weights are resident, the KV cache is the served cache). Per token per layer, int8: the four linears'
  // input widths (LK[j]) plus attention's Q ((HQ/tp)·HD); the lm_head input once. Independent of B and of L (the cache
  // read isn't re-stored); it scales with the sub-round's tokens (R.dp.T) and so is the cost of a larger T.
  const double store_tok_layer = (double)R.LK[0] + R.LK[1] + R.LK[2] + R.LK[3] + (double)(T_HQ / R.tp) * T_HD;
  const double defer_store = (store_tok_layer * T_LAYERS + (double)T_HID) * (double)R.dp.T;
  // the deferred demand a step = the step's own served MACs (each clean step makes one step's noised products,
  // batched over the sub-round); the chunk quota rounds that to whole chunks
  if (R.dp.on) {
    const unsigned long long dpairs = (unsigned long long)(served / 16384.0 + 0.5);  // served MACs / (4096 MAC/MMA · 4 MMA/pair)
    R.dp.dcap = (int)((dpairs + R.dp.chunk_pairs() - 1) / R.dp.chunk_pairs());
  }
  const char* dflt = R.dp.on ? "base defer dalone" : R.pf.on ? "base phase nocomm comm phase base" : "base nocomm comm base";
  std::string arms = getenv("TPC_ARMS") ? getenv("TPC_ARMS") : dflt;
  bool wrapped = false;
  int rc = 0;
  size_t pos = 0;
  while (pos < arms.size()) {
    size_t e = arms.find_first_of(" ,", pos);
    if (e == std::string::npos) e = arms.size();
    const std::string a = arms.substr(pos, e - pos);
    pos = e + 1;
    if (a.empty()) continue;
    const int kind = a == "nocomm" ? 1 : a == "comm" || a == "commphase" ? 2 : 0;
    const bool dalone = a == "dalone", defer = a == "defer";
    int sidework = (a == "phase" || a == "commphase") ? 1 : defer ? 2 : 0;
    if (sidework == 1 && !R.pf.on) continue;
    if ((defer || dalone) && !R.dp.on) continue;
    barrier(R);
    if (sidework) CK(cudaMemset(R.pf.total, 0, pf_total_bytes(R.pf)));
    if (defer || dalone) CK(cudaMemset(R.dp.dtotal, 0, 16));  // dtotal accumulates over the timed steps (dprolog resets only dctr)
    const std::vector<Rep> v = dalone ? time_work(R.d, [&] { enqueue_dalone(R); }, reps)
                                      : time_work(R.d, [&] { enqueue_step(R, p, lm, kind, sidework, B, L, &wrapped); }, reps);
    double mmas_rep = 0;
    char extra[1024] = "";
    if (sidework == 1) {
      unsigned long long tot[2], pairs_last, whole_last;
      pf_total_sum(R.pf, tot);  // the last step's pairs over every collective, not the sum over time_work's repeats
      mmas_rep = 4.0 * tot[0];
      long shape_bad, cut_warps;
      KCheck ck;
      pf_check(R, 8, &shape_bad, &cut_warps, &ck, &pairs_last, &whole_last);
      // the counters the fill is read from must equal the records' sums, as pf_gate and alone require of one launch
      if (tot[0] != pairs_last || tot[1] != whole_last) {
        fprintf(stderr, "rank %d %s B %d L %d: filler counters %llu/%llu != records %llu/%llu (pf_total_sum must cover "
                "every collective's slots)\n", R.r, a.c_str(), B, L, tot[0], tot[1], pairs_last, whole_last);
        shape_bad++;
      }
      snprintf(extra, sizeof extra, ", \"fill_mmas_full_rep\": %.1f, \"fill_mmas_last\": %.1f, \"rec_checked\": %ld, "
               "\"rec_bad\": %ld, \"whole_checked\": %ld, \"whole_bad\": %ld, \"whole_control_equal\": %ld, "
               "\"warps_checked\": %ld, \"warps_bad\": %ld, \"warps_control_equal\": %ld, \"shape_bad\": %ld, "
               "\"cut_warps\": %ld, \"pf\": [%d, %d, %d, %d, %d, %d, %d], "
               "\"fold_nc\": %d, \"fold_s\": %d, \"fold_xin\": %d",
               4.0 * tot[1], 4.0 * pairs_last, ck.whole + ck.warps,
               ck.whole_bad + ck.warps_bad + ck.whole_ctl + ck.warps_ctl, ck.whole, ck.whole_bad, ck.whole_ctl, ck.warps,
               ck.warps_bad, ck.warps_ctl, shape_bad, cut_warps, PF_W * KF_G, R.pf.NC, R.pf.S, R.pf.tiles, R.pf.tiles_g,
               R.pf.poll, R.pf.grid, R.pf.NC, R.pf.S, TP_XIN);
      if (!ck.clean() || shape_bad) rc = 8;
    } else if (defer || dalone) {
      unsigned long long dtot[2];
      CK(cudaMemcpy(dtot, R.dp.dtotal, 16, cudaMemcpyDeviceToHost));
      const double gap_pairs = (double)dtot[0] / (reps + 2), drain_pairs = (double)dtot[1] / (reps + 2);
      long fill_bad = 0;
      unsigned long long fpairs = 0;
      if (defer) {  // the filler fill left after the deferred quota took its share of the gaps
        unsigned long long ft[2], fwhole;
        long fcut;
        pf_total_sum(R.pf, ft);  // the last step's pairs over every collective
        mmas_rep = 4.0 * ft[0];
        const KRecs k = kf_fetch(R.pf, 0, R.pf.nslots);
        fill_bad = kf_shape(R.pf, k, 0, R.pf.nslots, &fcut, &fpairs, &fwhole);
        if (ft[0] != fpairs || ft[1] != fwhole) {
          fprintf(stderr, "rank %d defer B %d L %d: filler counters %llu/%llu != records %llu/%llu\n", R.r, B, L, ft[0],
                  ft[1], fpairs, fwhole);
          fill_bad++;
        }
      }
      const DCheck dc = dp_sample(R, 8);
      snprintf(extra, sizeof extra, ", \"defer_gap_pairs_rep\": %.1f, \"defer_drain_pairs_rep\": %.1f, "
               "\"defer_gap_macs_rep\": %.6e, \"defer_drain_macs_rep\": %.6e, \"defer_demand_macs\": %.6e, \"dcap\": %d, "
               "\"dchunk_pairs\": %lld, \"dSd\": %d, \"dT\": %d, \"dwrap\": %d, \"dmaxc\": %d, \"dgrid\": %d, "
               "\"defer_store_bytes\": %.6e, \"fold_nc\": %d, \"fold_s\": %d, \"fold_xin\": %d, "
               "\"dp_whole_checked\": %ld, \"dp_whole_bad\": %ld, \"dp_whole_control_equal\": %ld",
               gap_pairs, drain_pairs, gap_pairs * 4 * 4096.0, drain_pairs * 4 * 4096.0, served, R.dp.dcap,
               R.dp.chunk_pairs(), R.dp.Sd, R.dp.T, R.dp.dwrap, R.dp.dmaxc, defer ? R.dp.grid : R.dp.dgrid,
               defer_store, R.pf.NC, R.pf.S, TP_XIN, dc.whole, dc.whole_bad, dc.whole_ctl);
      if (defer) {
        const size_t used = strlen(extra);
        snprintf(extra + used, sizeof extra - used, ", \"fill_mmas_last\": %.1f, \"fill_shape_bad\": %ld", 4.0 * fpairs,
                 fill_bad);
      }
      if (!dc.clean() || fill_bad) rc = 8;
    }
    emit(R, a.c_str(), B, L, v, (dalone || kind == 2) ? 0 : served, mmas_rep, extra);
  }
  printf("{\"mode\": \"point\", \"tp\": %d, \"rank\": %d, \"B\": %d, \"L\": %d, \"kv_wrapped\": %s, \"lin_macs\": %.6e, "
         "\"att_macs\": %.6e, \"algos\": [%d, %d, %d, %d, %d], \"lt_ms\": [%.4f, %.4f, %.4f, %.4f, %.4f]}\n",
         R.tp, R.r, B, L, wrapped ? "true" : "false", lin, att, p[0].algo_id, p[1].algo_id, p[2].algo_id, p[3].algo_id,
         lm.algo_id, p[0].ms, p[1].ms, p[2].ms, p[3].ms, lm.ms);
  fflush(stdout);
  return rc;
}

static int pf_setup(PFill& f) {
  f.S = getenv_i("TPC_S", 64);
  f.tiles = getenv_i("TPC_TILES", 2);
  f.tiles_g = getenv_i("TPC_TILES_G", 16);
  f.poll = getenv_i("TPC_POLL", KF_PJ);
  f.NC = TP_NC;  // the fold is compile-time (TP_NC): duo (2, binding proved up to four words) by default, or ram/XOR-in (4, -DPFILL_XIN)
  if (f.S < 32 || f.S % 32) { fprintf(stderr, "TPC_S must be a positive multiple of 32 (pfill's)\n"); return 1; }
  if (f.poll > 0 && f.poll % KF_PJ) { fprintf(stderr, "TPC_POLL must be -1, 0 or a multiple of %d\n", KF_PJ); return 1; }
  if (f.tiles < 1 || f.tiles_g < 1 || f.grid < 1) { fprintf(stderr, "TPC_TILES, TPC_TILES_G and the grid must be >= 1\n"); return 1; }
  CK(cudaMalloc(&f.flags, f.nslots * sizeof(int)));
  CK(cudaMemset(f.flags, 0, f.nslots * sizeof(int)));
  CK(cudaMalloc(&f.epoch, 8));
  CK(cudaMemset(f.epoch, 0, 8));
  CK(cudaMalloc(&f.total, pf_total_bytes(f)));
  CK(cudaMemset(f.total, 0, pf_total_bytes(f)));
  const size_t nr = (size_t)f.nslots * f.stride() * PF_W;
  CK(cudaMalloc(&f.rec, nr * 8));
  CK(cudaMalloc(&f.qrec, nr * 4));
  CK(cudaMalloc(&f.cnt, (size_t)f.nslots * f.groups() * 4));
  CK(cudaStreamCreateWithFlags(&f.side, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&f.fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&f.join, cudaEventDisableTiming));
  CK(cudaFuncSetAttribute(kfill_k<TP_NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)KF_SMEM));
  CK(cudaFuncSetAttribute(kfill_k<TP_NC>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  // loaded now: under lazy loading a kernel's first launch can wait for the running kernels, so the cut gate's flag
  // would rise only after the filler it should cut had finished
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, raise_after_k));
  CK(cudaFuncGetAttributes(&fa, glue_k));
  CK(cudaFuncGetAttributes(&fa, prolog_k));
  f.on = true;
  return 0;
}

// the deferred pass's operands, buffers and grids (after pf_setup, so R.pf.grid is set). TPC_DT is the sub-round's tokens;
// Sd = TPC_DT / (PF_W · 16) fragments a warp a chunk. The operand footprint (TPC_DBUF_MB, default 256) exceeds L2, so the
// loads are real DRAM.
static int dp_setup(Rank& R) {
  DPass& dp = R.dp;
  dp.T = getenv_i("TPC_DT", 1024);
  if (dp.T % (PF_W * 16) || dp.T / (PF_W * 16) % 4 || dp.T < PF_W * 16 * 4) {
    fprintf(stderr, "TPC_DT must be a multiple of %d with TPC_DT/%d a positive multiple of 4 (1024, 4096, ...)\n", PF_W * 16, PF_W * 16);
    return 1;
  }
  dp.Sd = dp.T / (PF_W * 16);
  dp.grid = R.pf.grid;                                     // beside a collective: the filler's grid (sms - reserve)
  dp.dgrid = R.d.sms - getenv_i("TPC_DRAIN_RESERVE", 0);   // the drain and dalone: every SM
  dp.dmaxc = std::max(1, getenv_i("TPC_DMAXC", 4));        // chunks a group a beside-collective launch (bounds the hog)
  const size_t perA = (size_t)PF_W * dp.Sd * 512, perB = (size_t)PF_TN * 32;
  const size_t budget = (size_t)getenv_i("TPC_DBUF_MB", 256) << 20;
  dp.dwrap = std::max(256, (int)(budget / (2 * perA + 2 * perB)));
  CK(cudaMalloc(&dp.dA, (size_t)dp.dwrap * perA));
  CK(cudaMalloc(&dp.dU, (size_t)dp.dwrap * perA));
  CK(cudaMalloc(&dp.dB, (size_t)dp.dwrap * perB));
  CK(cudaMalloc(&dp.dBias, (size_t)dp.dwrap * perB));
  fill_s8<<<4096, 256, 0, R.d.st>>>(dp.dA, (size_t)dp.dwrap * perA, TP_SA);
  fill_s8<<<4096, 256, 0, R.d.st>>>(dp.dU, (size_t)dp.dwrap * perA, TP_SU);
  fill_s8<<<4096, 256, 0, R.d.st>>>(dp.dB, (size_t)dp.dwrap * perB, TP_SB);
  fill_s8<<<4096, 256, 0, R.d.st>>>(dp.dBias, (size_t)dp.dwrap * perB, TP_SBIAS);
  CK(cudaMalloc(&dp.dctr, 4));
  CK(cudaMalloc(&dp.dtotal, 16));
  CK(cudaMalloc(&dp.drec, (size_t)TP_DR_MAX * PF_W * 8));
  CK(cudaMalloc(&dp.dqrec, (size_t)TP_DR_MAX * PF_W * 4));
  CK(cudaFuncSetAttribute(tp_dpass_k<TP_NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)KF_SMEM));
  CK(cudaFuncSetAttribute(tp_dpass_k<TP_NC>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(tp_drain_k<TP_NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)KF_SMEM));
  CK(cudaFuncSetAttribute(tp_drain_k<TP_NC>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, dprolog_k));
  CK(cudaStreamSynchronize(R.d.st));
  CK(cudaGetLastError());
  dp.on = true;
  return 0;
}

// the deferred pass's gate before timing: drain TPC_DP_GATE (8) whole chunks on one block, each warp receipt against
// tp_defer_ref_warp (and not its Sd - 4 control), every warp at q = full, and the pairs counter exact.
static int dp_gate(Rank& R) {
  DPass& dp = R.dp;
  const int cap = std::min(getenv_i("TPC_DP_GATE", 8), TP_DR_MAX);
  CK(cudaMemset(dp.dctr, 0, 4));
  CK(cudaMemset(dp.dtotal, 0, 16));
  CK(cudaMemset(dp.drec, 0xA5, (size_t)cap * PF_W * 8));
  CK(cudaMemset(dp.dqrec, 0xA5, (size_t)cap * PF_W * 4));
  tp_drain_k<TP_NC><<<1, 32 * PF_W * KF_G, KF_SMEM, R.d.st>>>(g_tc0, g_tc1, dp.Sd, dp.dA, dp.dB, dp.dU, dp.dBias, dp.dwrap,
                                                          dp.dctr, cap, dp.drec, dp.dqrec, dp.dtotal);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<unsigned long long> rec((size_t)cap * PF_W);
  std::vector<int> q((size_t)cap * PF_W);
  CK(cudaMemcpy(rec.data(), dp.drec, rec.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(q.data(), dp.dqrec, q.size() * 4, cudaMemcpyDeviceToHost));
  unsigned long long tot[2];
  CK(cudaMemcpy(tot, dp.dtotal, 16, cudaMemcpyDeviceToHost));
  const int full = dp.fullp();
  long whole = 0, bad = 0, ctl = 0, qbad = 0;
  for (int idx = 0; idx < cap; idx++)
    for (int w = 0; w < PF_W; w++) {
      qbad += q[(size_t)idx * PF_W + w] != full;
      const int wc = idx % dp.dwrap;
      const unsigned long long x = rec[(size_t)idx * PF_W + w];
      bad += x != tp_defer_ref_warp(g_tc0, g_tc1, (uint64_t)idx, wc, w, dp.Sd, dp.Sd, full);
      ctl += x == tp_defer_ref_warp(g_tc0, g_tc1, (uint64_t)idx, wc, w, dp.Sd, dp.Sd - 4, full);
      whole++;
    }
  const bool ok = whole > 0 && !bad && !ctl && !qbad && tot[1] == (unsigned long long)cap * PF_W * full && tot[0] == 0;
  printf("{\"mode\": \"dp_gate\", \"rank\": %d, \"Sd\": %d, \"T\": %d, \"dwrap\": %d, \"cap\": %d, \"whole\": %ld, "
         "\"bad\": %ld, \"control_equal\": %ld, \"q_bad\": %ld, \"pairs\": %llu, \"ok\": %s}\n",
         R.r, dp.Sd, dp.T, dp.dwrap, cap, whole, bad, ctl, qbad, tot[1], ok ? "true" : "false");
  fflush(stdout);
  return ok ? 0 : 1;
}

// tpcomm alone [reps]: the filler alone on device 0, fixed work (TPC_ALONE_TILES tiles a group, default 8) on SMs -
// TPC_RESERVE blocks, each launch timed; its rate of the whole GPU's int8 peak and of its SMs'. ncu runs this mode for the
// DRAM gate (kernel kfill_k), against `tpcomm bw 1 3` (dcap's bw_read, 1 GiB a launch) as the counter's control.
static int alone_main(int argc, char** argv) {
  Rank R;
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&R.d.st));
  CK(cudaDeviceGetAttribute(&R.d.sms, cudaDevAttrMultiProcessorCount, 0));
  if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); return 2; }
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
  if (nvmlDeviceGetHandleByPciBusId_v2(bus, &R.d.nv) != NVML_SUCCESS) { fprintf(stderr, "NVML handle failed\n"); return 2; }
  PFill& f = R.pf;
  f.grid = R.d.sms - getenv_i("TPC_RESERVE", 8);
  f.nslots = 1;
  setenv("TPC_TILES", std::to_string(getenv_i("TPC_ALONE_TILES", 8)).c_str(), 1);
  setenv("TPC_TILES_G", std::to_string(getenv_i("TPC_ALONE_TILES", 8)).c_str(), 1);
  setenv("TPC_POLL", "-1", 1);
  if (pf_setup(f)) return 1;
  const int reps = argc > 2 ? atoi(argv[2]) : 9;
  const std::vector<Rep> v = time_work(R.d, [&] { kfill_launch(f, 0, g_tc0, g_tc1, R.d.st); }, reps);
  unsigned long long ep, tot[2];
  CK(cudaMemcpy(&ep, f.epoch, sizeof ep, cudaMemcpyDeviceToHost));
  pf_total_sum(f, tot);
  const KRecs k = kf_fetch(f, 0, 1);
  long partial;
  unsigned long long pairs, whole;
  const long shape_bad = kf_shape(f, k, 0, 1, &partial, &pairs, &whole);
  std::mt19937_64 rng(0xA10Eull);
  const KCheck ck = kf_sample(f, k, 0, 1, ep, 4, rng);
  std::vector<Rep> s = v;
  std::sort(s.begin(), s.end(), [](const Rep& a, const Rep& b) { return a.ms < b.ms; });
  const Rep& m = s[s.size() / 2];
  const double macs = (double)f.groups() * f.tiles * PF_W * f.full() * 4 * 4096.0;
  const double peak = 1024.0 * R.d.sms * m.mhz * 1e6;
  const bool ok = ck.clean() && ck.whole > 0 && !shape_bad && !partial && pairs == (unsigned long long)f.groups() * f.tiles * PF_W * f.full() &&
                  tot[0] == pairs;  // one launch's pairs (per-warp overwrite), not the sum over the repeats
  printf("{\"mode\": \"alone\", \"grid\": %d, \"groups\": %lld, \"S\": %d, \"tiles\": %d, \"ms\": %.4f, \"ms_min\": %.4f, "
         "\"ms_max\": %.4f, \"mhz\": %.1f, \"watt\": %.1f, \"macs\": %.6e, \"share_gpu\": %.4f, \"share_fill_sms\": %.4f, "
         "\"whole_checked\": %ld, \"whole_bad\": %ld, \"whole_control_equal\": %ld, \"shape_bad\": %ld, \"ok\": %s}\n",
         f.grid, f.groups(), f.S, f.tiles, m.ms, s.front().ms, s.back().ms, m.mhz, m.watt, macs, macs / (m.ms * 1e-3) / peak,
         macs / (m.ms * 1e-3) / peak * R.d.sms / f.grid, ck.whole, ck.whole_bad, ck.whole_ctl, shape_bad, ok ? "true" : "false");
  fflush(stdout);
  return ok ? 0 : 8;
}

static int rank_main(int r, int tp, const ncclUniqueId& id, int argc, char** argv) {
  Rank R;
  R.r = r;
  R.tp = tp;
  CK(cudaSetDevice(r));
  CK(cudaStreamCreate(&R.d.st));
  CK(cudaDeviceGetAttribute(&R.d.sms, cudaDevAttrMultiProcessorCount, r));
  if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); return 2; }
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, r));
  if (nvmlDeviceGetHandleByPciBusId_v2(bus, &R.d.nv) != NVML_SUCCESS) { fprintf(stderr, "NVML handle failed\n"); return 2; }
  char uuid[96] = {0};
  nvmlDeviceGetUUID(R.d.nv, uuid, sizeof uuid);
  if (const char* want = getenv("GPU_LEASE_UUID")) {
    if (*want && !strstr(want, uuid)) { fprintf(stderr, "rank %d: device %s is not leased (%s)\n", r, uuid, want); return 3; }
  }
  fprintf(stderr, "rank %d: device %s bus %s, %d SMs\n", r, uuid, bus, R.d.sms);
  NK(ncclCommInitRank(&R.comm, tp, id, r));
  R.LN[0] = T_QKV / tp; R.LK[0] = T_HID;
  R.LN[1] = T_HID;      R.LK[1] = T_HID / tp;
  R.LN[2] = 2 * T_FF / tp; R.LK[2] = T_HID;
  R.LN[3] = T_HID;      R.LK[3] = T_FF / tp;
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  CK(cudaMalloc(&g_ref, (size_t)256 * VOCAB * sizeof(int)));
  CK(cudaMalloc(&g_cnt, sizeof *g_cnt));
  for (int l = 0; l < T_LAYERS; l++)
    for (int j = 0; j < 4; j++) {
      const size_t n = (size_t)R.LN[j] * R.LK[j];
      CK(cudaMalloc(&R.W[l][j], n));
      fill_s8<<<1024, 256, 0, R.d.st>>>(R.W[l][j], n, 0x1000 + l * 4 + j + 977 * r);
    }
  CK(cudaMalloc(&R.Wlm, (size_t)(T_VOCAB / tp) * T_HID));
  fill_s8<<<1024, 256, 0, R.d.st>>>(R.Wlm, (size_t)(T_VOCAB / tp) * T_HID, 0x77 + r);
  const int maxB = 256, maxL = getenv_i("TPC_MAXL", 8192);
  CK(cudaMalloc(&R.X, (size_t)maxB * T_FF));
  fill_s8<<<1024, 256, 0, R.d.st>>>(R.X, (size_t)maxB * T_FF, 0x55);
  CK(cudaMalloc(&R.D, (size_t)maxB * (T_VOCAB / tp + 64) * 4 + ((size_t)maxB * 2 * T_FF / tp) * 4));
  CK(cudaMalloc(&R.ar, (size_t)maxB * T_HID * 2));
  CK(cudaMemset(R.ar, 0, (size_t)maxB * T_HID * 2));
  CK(cudaMalloc(&R.logits, (size_t)maxB * (T_VOCAB / tp) * 2));
  CK(cudaMalloc(&R.gath, (size_t)maxB * T_VOCAB * 2));
  const size_t maxBe = maxB / tp;
  CK(cudaMalloc(&R.q, std::max<size_t>((size_t)maxB * T_QKV, maxBe * HQ * HD)));
  fill_s8<<<64, 256, 0, R.d.st>>>(R.q, maxBe * HQ * HD, 0x31);
  CK(cudaMalloc(&R.P, maxBe * HQ * maxL));
  fill_s8<<<1024, 256, 0, R.d.st>>>(R.P, maxBe * HQ * maxL, 0x32);
  CK(cudaMalloc(&R.S, maxBe * HQ * maxL * sizeof(int)));
  CK(cudaMalloc(&R.O, std::max<size_t>(maxBe * HQ * HD, (size_t)maxB * T_HID / tp) * sizeof(int)));
  CK(cudaMemset(R.O, 0, std::max<size_t>(maxBe * HQ * HD, (size_t)maxB * T_HID / tp) * sizeof(int)));
  // the phase filler
  PFill& f = R.pf;
  f.on = getenv_i("TPC_PHASE", 1) != 0;
  f.grid = R.d.sms - getenv_i("TPC_RESERVE", 8);
  f.nslots = 2 * T_LAYERS + 2;
  if (f.on && pf_setup(f)) return 1;
  // the deferred pass (needs the filler's grid), before the KV pool so its operands fit
  if (getenv_i("TPC_DEFER", 1) && f.on && dp_setup(R)) return 1;
  // the KV pool: what is left, less room for cuBLASLt and NCCL
  size_t fr, tot;
  CK(cudaMemGetInfo(&fr, &tot));
  const size_t keep = (size_t)getenv_i("TPC_KEEP_GB", 4) << 30;
  R.kv_bytes = std::min((size_t)getenv_i("TPC_KV_GB", 40) << 30, fr > keep ? fr - keep : 0);
  CK(cudaMalloc(&R.kv, R.kv_bytes));
  fill_s8<<<4096, 256, 0, R.d.st>>>(R.kv, R.kv_bytes, 0x99);
  CK(cudaStreamSynchronize(R.d.st));
  CK(cudaGetLastError());
  fprintf(stderr, "rank %d: weights %.2f GB, KV pool %.2f GB\n", r,
          ((double)T_LAYERS * (R.LN[0] * (double)R.LK[0] + R.LN[1] * (double)R.LK[1] + R.LN[2] * (double)R.LK[2] +
                               R.LN[3] * (double)R.LK[3]) + (double)(T_VOCAB / tp) * T_HID) / 1e9,
          R.kv_bytes / 1e9);
  int rc = 0;
  if (f.on) {
    const int g = pf_gate(R);
    if (g) {
      fprintf(stderr, "rank %d: phase filler gate failed (%d)\n", r, g);
      rc = 5;
    }
  }
  if (R.dp.on) {
    const int g = dp_gate(R);
    if (g) {
      fprintf(stderr, "rank %d: deferred pass gate failed (%d)\n", r, g);
      rc = rc ? rc : 7;
    }
  }
  // every rank agrees on the gate before anything is timed
  float ok = rc ? 1.f : 0.f, *dok;
  CK(cudaMalloc(&dok, 4));
  CK(cudaMemcpy(dok, &ok, 4, cudaMemcpyHostToDevice));
  NK(ncclAllReduce(dok, dok, 1, ncclFloat, ncclSum, R.comm, R.d.st));
  CK(cudaMemcpy(&ok, dok, 4, cudaMemcpyDeviceToHost));
  if (ok != 0.f) { ncclCommDestroy(R.comm); return rc ? rc : 5; }
  const int reps = argc > 3 ? atoi(argv[3]) : 9;
  std::string pts = argv[2];
  for (size_t pos = 0; pos < pts.size();) {
    size_t e = pts.find(' ', pos);
    if (e == std::string::npos) e = pts.size();
    const std::string bl = pts.substr(pos, e - pos);
    pos = e + 1;
    if (bl.empty()) continue;
    const int B = atoi(bl.c_str()), L = atoi(bl.substr(bl.find(':') + 1).c_str());
    if (L > maxL) { fprintf(stderr, "L %d over TPC_MAXL %d\n", L, maxL); rc = 1; continue; }
    const int prc = run_point(R, B, L, reps);
    if (prc) rc = prc;
  }
  NK(ncclCommDestroy(R.comm));
  return rc;
}

// The launcher: one rank a visible GPU (or TPC_TP of them), each this binary re-executed with TPC_RANK and the NCCL id,
// its stdout to $TPC_OUT/rank<r>.jsonl and stderr to rank<r>.err when TPC_OUT is set.
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "bw") return dcap_main(argc, argv);
  if ((mode != "run" || argc < 3) && mode != "alone") {
    fprintf(stderr, "usage: tpcomm run \"B:L ...\" [reps]   (TP = the visible GPUs, or TPC_TP; env: TPC_PHASE, TPC_S,\n"
                    "       TPC_TILES, TPC_TILES_G, TPC_POLL, TPC_RESERVE, TPC_CUT_CYC, TPC_ARMS, TPC_MAXL, TPC_KV_GB,\n"
                    "       DFILL_COINS)\n"
                    "       tpcomm alone [reps]   (the filler alone on device 0: TPC_ALONE_TILES, TPC_S, TPC_RESERVE)\n"
                    "       tpcomm bw GiB [reps]  (dcap's bw_read: the DRAM counter's control)\n");
    return 1;
  }
  if (const char* cs = getenv("DFILL_COINS")) {
    if (strlen(cs) != 32 || sscanf(cs, "%16llx%16llx", (unsigned long long*)&g_tc0, (unsigned long long*)&g_tc1) != 2) {
      fprintf(stderr, "DFILL_COINS must be 32 hex digits\n");
      return 1;
    }
  }
  if (mode == "alone") return alone_main(argc, argv);
  if (const char* rk = getenv("TPC_RANK")) {
    ncclUniqueId id;
    const char* hx = getenv("TPC_ID");
    if (!hx || strlen(hx) != 2 * sizeof id.internal) { fprintf(stderr, "TPC_ID missing\n"); return 1; }
    for (size_t i = 0; i < sizeof id.internal; i++) {
      unsigned v;
      sscanf(hx + 2 * i, "%2x", &v);
      id.internal[i] = (char)v;
    }
    return rank_main(atoi(rk), atoi(getenv("TPC_NRANKS")), id, argc, argv);
  }
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  const int tp = getenv_i("TPC_TP", ndev);
  if (tp < 1 || tp > ndev) { fprintf(stderr, "TP %d with %d visible GPUs\n", tp, ndev); return 1; }
  ncclUniqueId id;
  NK(ncclGetUniqueId(&id));
  char hx[2 * sizeof id.internal + 1];
  for (size_t i = 0; i < sizeof id.internal; i++) snprintf(hx + 2 * i, 3, "%02x", (unsigned char)id.internal[i]);
  const char* out = getenv("TPC_OUT");
  std::vector<pid_t> kids;
  for (int r = 0; r < tp; r++) {
    const pid_t pid = fork();
    if (pid == 0) {
      setenv("TPC_RANK", std::to_string(r).c_str(), 1);
      setenv("TPC_NRANKS", std::to_string(tp).c_str(), 1);
      setenv("TPC_ID", hx, 1);
      if (out) {
        const std::string o = std::string(out) + "/rank" + std::to_string(r);
        if (!freopen((o + ".jsonl").c_str(), "a", stdout) || !freopen((o + ".err").c_str(), "a", stderr)) _exit(9);
      }
      execv("/proc/self/exe", argv);
      _exit(10);
    }
    kids.push_back(pid);
  }
  int rc = 0;
  for (pid_t k : kids) {
    int st = 0;
    waitpid(k, &st, 0);
    const int c = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    if (c) rc = c;
  }
  return rc;
}
