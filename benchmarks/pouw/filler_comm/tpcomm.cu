// Idle tensor time in tensor-parallel decode over PCIe, and the DRAM-free filler inside it (HC-COMM, cmp-13).
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
// The phase filler is pfill-k (HC-PROVED, cursor/hc-proved-ea24@fe2fdb5bf, pfill.cuh built with PFILL_BITS): Theorem K's
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
#define main dcap_main
#include "../decode_capacity/dcap.cu"
#undef main

#define PFILL_BITS
#define DFill PfDFill
#define dfill_k pf_dfill_k
#define dfill_launch pf_dfill_launch
#define dfill_prep pf_dfill_prep
#define dfill_make pf_dfill_make
#define dfill_check pf_dfill_check
#define dfill_ref pf_dfill_ref
#include "../decode_capacity/pfill.cuh"
#undef DFill
#undef dfill_k
#undef dfill_launch
#undef dfill_prep
#undef dfill_make
#undef dfill_check
#undef dfill_ref

#include <nccl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <map>

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
constexpr size_t KF_SMEM_GRP = (size_t)PF_TN * 32 + (size_t)PF_W * 32 * 16 * 4;  // B as s8 bytes, A's 32 SHA blocks a warp
constexpr size_t KF_SMEM = KF_G * KF_SMEM_GRP;
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
// whole. total[0] sums the pairs every warp ran, total[1] the pairs of whole tiles alone.
template <int NC>
__global__ void __launch_bounds__(32 * PF_W * KF_G, 1)
    kfill_k(uint64_t c0, uint64_t c1, const unsigned long long* __restrict__ epoch, int slot, int nslots, int tiles, int maxcap,
            int S, int poll, const int* flag, unsigned long long* __restrict__ rec, int* __restrict__ qrec,
            int* __restrict__ cnt, unsigned long long* total) {
  extern __shared__ __align__(16) uint8_t kf_smem[];
  constexpr int P = 32;
  constexpr uint64_t dom = PF_DOM01;
  const int grp = threadIdx.x / (32 * PF_W), bar = 1 + grp;
  uint8_t* Bs = kf_smem + grp * KF_SMEM_GRP;
  uint32_t(*Ab)[32 * 16] = reinterpret_cast<uint32_t(*)[32 * 16]>(Bs + PF_TN * 32);
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
    PfKeys key{0, 0, 0, 0};
    if (lane <= 8) {
      uint64_t st[8];
      const int bb = w * 8 + (lane & 7);
      pf_block(dom, c0, c1, tile, lane < 8 ? PF_B + (uint64_t)bb : PF_KEY, st);
      if (lane < 8) {
#pragma unroll
        for (int nn = 0; nn < 16; nn++) {
          const uint32_t R = pf_row(st, nn);
          uint4* p = (uint4*)&Bs[(16 * bb + nn) * 32];
          p[0] = make_uint4(pf_01(R & 15), pf_01((R >> 16) & 15), pf_01((R >> 4) & 15), pf_01((R >> 20) & 15));
          p[1] = make_uint4(pf_01((R >> 8) & 15), pf_01((R >> 24) & 15), pf_01((R >> 12) & 15), pf_01(R >> 28));
        }
      } else {
        key = pf_keys(st);
      }
    }
    key.a0 = __shfl_sync(~0u, key.a0, 8);
    key.a1 = __shfl_sync(~0u, key.a1, 8);
    key.k0 = __shfl_sync(~0u, key.k0, 8);
    key.k1 = __shfl_sync(~0u, key.k1, 8);
    kf_bar(bar);
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
    int f0 = 0, next = (w >> 2) & 1 ? P / 2 : P;
    agen(0);
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
    if (lane == 0) {
      rec[(rbase + i) * PF_W + w] = h;
      qrec[(rbase + i) * PF_W + w] = q;
    }
    mine += q;
    qlast = q;
  }
  const int part = kf_bar_or(bar, started > 0 && qlast < full);
  if (lane == 0) atomicAdd(total, mine);
  if (threadIdx.x % (32 * PF_W) == 0) {
    cnt[(size_t)slot * ng + gid] = started;
    atomicAdd(total + 1, (unsigned long long)(started - part) * PF_W * full);
  }
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
          const uint64_t b0 = Bb[8 * j + 2 * t], b1 = Bb[8 * j + 2 * t + 1];
          pf_absorb<4>(y, z, pf_word<PF_BITS>(R0, b0), pf_word<PF_BITS>(R0, b1), pf_word<PF_BITS>(R1, b0),
                       pf_word<PF_BITS>(R1, b1), key);
        }
      }
      h ^= pf_fold(K, pos, (uint64_t)y);
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
  kfill_k<4><<<f.grid, 32 * PF_W * KF_G, KF_SMEM, st>>>(c0, c1, f.epoch, slot, f.nslots, f.cap(slot), f.maxcap(), f.S, f.poll,
                                                        f.flags + slot, f.rec, f.qrec, f.cnt, f.total);
}

static uint64_t g_tc0 = 0x0123456789abcdefull, g_tc1 = 0xfedcba9876543210ull;  // the round's coins (DFILL_COINS)

// a collective at slot `slot`: the phase filler forks beside it when `pf`; `fn` enqueues the NCCL call on st
static void collective(Rank& R, bool pf, int slot, cudaStream_t st, const std::function<void()>& fn) {
  if (pf) {
    CK(cudaEventRecord(R.pf.fork, st));
    CK(cudaStreamWaitEvent(R.pf.side, R.pf.fork, 0));
    kfill_launch(R.pf, slot, g_tc0, g_tc1, R.pf.side);
  }
  fn();
}

// kind: 0 the step, 1 without collectives, 2 the collectives alone; pf: the phase filler beside each collective
static void enqueue_step(Rank& R, const std::vector<Lt>& p, const Lt& lm, int kind, bool pf, int B, int L, bool* wrapped) {
  cudaStream_t st = R.d.st;
  const size_t hb = (size_t)B * T_HID * 2;  // the BF16 hidden state
  const int Be = B / R.tp;
  const bool comp = kind != 2, comm = kind != 1;
  int slot = 0;
  int* flag = nullptr;  // the flag the next small kernel raises
  auto coll_ar = [&] {
    if (!comm) return;
    collective(R, pf, slot, st, [&] { NK(ncclAllReduce(R.ar, R.ar, (size_t)B * T_HID, ncclBfloat16, ncclSum, R.comm, st)); });
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
      r.whole_bad += x != pf_dfill_ref(g_tc0, g_tc1, tile, 4, f.S);
      r.whole_ctl += x == pf_dfill_ref(g_tc0, g_tc1, tile, 4, f.S - 8);
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
  for (const char* kind : {"fixed", "up", "cut"}) {
    const std::string kd = kind;
    f.poll = kd == "fixed" ? -1 : KF_PJ;
    CK(cudaMemset(f.rec + f.rix(slot, 0, 0, 0), 0xA5, nr * 8));
    CK(cudaMemset(f.qrec + f.rix(slot, 0, 0, 0), 0xA5, nr * 4));
    CK(cudaMemset(f.cnt + (size_t)slot * f.groups(), 0xA5, f.groups() * 4));
    CK(cudaMemset(f.total, 0, 16));
    CK(cudaMemset(f.flags, 0, f.nslots * sizeof(int)));
    if (kd == "up") CK(cudaMemset(f.flags + slot, 1, 1));
    CK(cudaDeviceSynchronize());
    kfill_launch(f, slot, g_tc0, g_tc1, R.pf.side);
    if (kd == "cut") raise_after_k<<<1, 32, 0, R.d.st>>>(f.flags + slot, getenv_i("TPC_CUT_CYC", 200000));
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    const KRecs k = kf_fetch(f, slot, 1);
    unsigned long long tot[2];
    CK(cudaMemcpy(tot, f.total, 16, cudaMemcpyDeviceToHost));
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
            ck.whole_bad += x != pf_dfill_ref(g_tc0, g_tc1, tile, 4, f.S);
            ck.whole_ctl += x == pf_dfill_ref(g_tc0, g_tc1, tile, 4, f.S - 8);
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
static void pf_check(Rank& R, int n, long* shape_bad, long* cut_warps, KCheck* ck, unsigned long long* pairs_last) {
  PFill& f = R.pf;
  unsigned long long ep, whole;
  CK(cudaMemcpy(&ep, f.epoch, sizeof ep, cudaMemcpyDeviceToHost));
  const KRecs k = kf_fetch(f, 0, f.nslots);
  *shape_bad = kf_shape(f, k, 0, f.nslots, cut_warps, pairs_last, &whole);
  std::mt19937_64 rng(ep * 7919 + R.r);
  *ck = kf_sample(f, k, 0, f.nslots, ep, n, rng);
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
  std::string arms = getenv("TPC_ARMS") ? getenv("TPC_ARMS") : (R.pf.on ? "base phase nocomm comm phase base" : "base nocomm comm base");
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
    const bool pf = a == "phase" || a == "commphase";
    if (pf && !R.pf.on) continue;
    barrier(R);
    if (pf) CK(cudaMemset(R.pf.total, 0, 16));
    const std::vector<Rep> v = time_work(R.d, [&] { enqueue_step(R, p, lm, kind, pf, B, L, &wrapped); }, reps);
    double mmas_rep = 0;
    char extra[512] = "";
    if (pf) {
      unsigned long long tot[2], pairs_last;
      CK(cudaMemcpy(tot, R.pf.total, 16, cudaMemcpyDeviceToHost));
      mmas_rep = 4.0 * tot[0] / (reps + 2);  // time_work's 2 warm-ups ran it too
      long shape_bad, cut_warps;
      KCheck ck;
      pf_check(R, 8, &shape_bad, &cut_warps, &ck, &pairs_last);
      snprintf(extra, sizeof extra, ", \"fill_mmas_full_rep\": %.1f, \"fill_mmas_last\": %.1f, \"rec_checked\": %ld, "
               "\"rec_bad\": %ld, \"whole_checked\": %ld, \"whole_bad\": %ld, \"whole_control_equal\": %ld, "
               "\"warps_checked\": %ld, \"warps_bad\": %ld, \"warps_control_equal\": %ld, \"shape_bad\": %ld, "
               "\"cut_warps\": %ld, \"pf\": [%d, %d, %d, %d, %d, %d, %d]",
               4.0 * tot[1] / (reps + 2), 4.0 * pairs_last, ck.whole + ck.warps,
               ck.whole_bad + ck.warps_bad + ck.whole_ctl + ck.warps_ctl, ck.whole, ck.whole_bad, ck.whole_ctl, ck.warps,
               ck.warps_bad, ck.warps_ctl, shape_bad, cut_warps, PF_W * KF_G, R.pf.NC, R.pf.S, R.pf.tiles, R.pf.tiles_g,
               R.pf.poll, R.pf.grid);
      if (!ck.clean() || shape_bad) rc = 8;
    }
    emit(R, a.c_str(), B, L, v, kind == 2 ? 0 : served, mmas_rep, extra);
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
  if (getenv_i("TPC_NC", 4) != 4) { fprintf(stderr, "pfill-k's receipt is the ram fold: TPC_NC 4\n"); return 1; }
  if (f.S < 32 || f.S % 32) { fprintf(stderr, "TPC_S must be a positive multiple of 32 (pfill's)\n"); return 1; }
  if (f.poll > 0 && f.poll % KF_PJ) { fprintf(stderr, "TPC_POLL must be -1, 0 or a multiple of %d\n", KF_PJ); return 1; }
  if (f.tiles < 1 || f.tiles_g < 1 || f.grid < 1) { fprintf(stderr, "TPC_TILES, TPC_TILES_G and the grid must be >= 1\n"); return 1; }
  CK(cudaMalloc(&f.flags, f.nslots * sizeof(int)));
  CK(cudaMemset(f.flags, 0, f.nslots * sizeof(int)));
  CK(cudaMalloc(&f.epoch, 8));
  CK(cudaMemset(f.epoch, 0, 8));
  CK(cudaMalloc(&f.total, 16));
  CK(cudaMemset(f.total, 0, 16));
  const size_t nr = (size_t)f.nslots * f.stride() * PF_W;
  CK(cudaMalloc(&f.rec, nr * 8));
  CK(cudaMalloc(&f.qrec, nr * 4));
  CK(cudaMalloc(&f.cnt, (size_t)f.nslots * f.groups() * 4));
  CK(cudaStreamCreateWithFlags(&f.side, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&f.fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&f.join, cudaEventDisableTiming));
  CK(cudaFuncSetAttribute(kfill_k<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)KF_SMEM));
  CK(cudaFuncSetAttribute(kfill_k<4>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  f.on = true;
  return 0;
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
  CK(cudaMemcpy(tot, f.total, 16, cudaMemcpyDeviceToHost));
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
                  tot[0] == pairs * (reps + 2);
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
