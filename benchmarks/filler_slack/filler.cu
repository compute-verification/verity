// The receipted INT8 filler on sm_120 and its issue slack ν.
//
// The filler is A5a's best fused loop at L = 32 (`benchmarks/receipt_phi/phi.cu` on cursor/receipt-phi-ea24,
// `fused_k<DUO, 1, s8>`): each warp runs FT = 4 m16n8k32 s8 x s8 -> s32 tiles over salted register operands, and after
// every MMA round (32 k-steps) each lane absorbs its 16 accumulator words into four duo-φ chains. One kernel template
// carries the three ways of measuring what the filler leaves free:
//
//   XOP, X, UNR   X extra integer instructions of class XOP per loop trip of UNR rounds (4·UNR MMAs) inside the
//                 filler's own loop, reading no accumulator (method b: k = X / (4·UNR) per MMA)
//   POP, PB       warps FW..FW+PW-1 of every block run a dependent-free stream of class POP, PB a body, with an optional
//                 __nanosleep between bodies, until the block's filler warps finish (method a, co-resident probe); POP =
//                 IMMA runs covert MMAs instead (the tensor time the filler leaves idle)
//   pk<POP, PB>   the same probe as its own kernel on a second stream, one block on every SM beside the plain filler's
//                 (base-u1's own binary), until that SM's filler warps finish (method a, second stream)
//
// One block per SM (it reserves the SM's shared memory). Filler warps are 0..FW-1, so with FW and PW multiples of 4
// every SMSP holds FW/4 filler and PW/4 probe warps. Modes:
//
//   filler gate SEED KSEED           small instances of every kernel: "G" lines for gate.py to recompute
//   filler sweep ITERS REPS          every configuration, interleaved rep by rep: JSON lines, "V" lines for gate.py
//   filler one NAME FW PW SLEEP ITERS   one launch of one kernel (for ncu)
//   filler list                      the kernel names
//
// Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 filler.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(3);                                                                             \
    }                                                                                      \
  } while (0)

constexpr int NKEY = 32;
__constant__ uint32_t KA[NKEY], KK[NKEY];

__host__ __device__ __forceinline__ uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__host__ __device__ __forceinline__ uint32_t gen(uint32_t seed, uint32_t a, uint32_t b, uint32_t c) {
  return lowbias32(seed ^ lowbias32((a * 0x9E3779B1U) ^ lowbias32((b * 0x85EBCA77U) ^ lowbias32(c + 0x632BE5ABU))));
}

__device__ __forceinline__ uint32_t rotl(uint32_t x, int r) { return __funnelshift_l(x, x, r); }
__host__ __device__ constexpr int rot_of(int t) { return ((t * 7 + 5) % 31) + 1; }

struct St {
  uint32_t y, z;
};

// duo: u = y ^ p ^ k_t; y <- u·a_t + z; z <- rotl(u, r_t) + y. Receipt y ^ rotl(z, 16).
template <int T0>
__device__ __forceinline__ void duo4(St& s, const uint32_t (&p)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int t = T0 + i;
    const uint32_t u = s.y ^ p[i] ^ KK[t];
    s.y = u * KA[t] + s.z;
    s.z = rotl(u, rot_of(t)) + s.y;
  }
}
__device__ __forceinline__ uint32_t finish(const St& s) { return s.y ^ rotl(s.z, 16); }

__device__ __forceinline__ void mma_s8(uint32_t (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The extra (method b) and probe (method a) instruction classes. Each op updates register u = e mod 8 from itself and
// register e + 3 mod 8, last written five ops earlier, so a run of ops is dependency-free at the ALU's latency and no two
// ops fold into one. MIX alternates IADD3 and IMAD, one ALU-pipe and one FMA-pipe op.
// IMMA (probes only) is a covert m16n8k32 s8 MMA into one of eight accumulators the probe warp holds: not issue slack
// but the tensor time the filler leaves idle, P1's falsifier.
// Round 1's probes (IADD3/LOP3/IMAD) read three vector registers each (SASS `IADD3 Rd, PT, PT, Ra, Rb, Rc`). Round 2
// adds the cheapest merge-capable forms, a form that folds one computed word (an addend) into a running sum (an
// accumulator), with as few vector-register reads as possible, since W10 (note:20261015T0704Z-draft-credit-ceilings)
// found the read count, not the instruction class, sets tensor-pipe contention. An immediate is not an addend.
//   MERGE  y[u] += y[v]                 IADD3 Rd, Rd, Rv, RZ          two vector reads
//   REUSE  y[u] += y[7] (y[7] fixed)    IADD3 Rd, Rd, R7.reuse, RZ    one fresh vector read (the addend from the reuse cache)
//   UNIF   y[u] += a warp-uniform word  IADD3 Rd, Rd, UR,  RZ         one vector read (addend in a uniform register)
//   CBANK  y[u] += KK[imm]              IADD3 Rd, Rd, c[0x3][..], RZ  one vector read (addend from the constant bank)
//   LOP3X  y[u] = byte-merge(y[u],y[v]) LOP3 Rd, Rd, Rv, imm, lut     two vector reads (a byte-packing merge)
//   IMADM  y[u] += c1·y[v] (c1 odd)     IMAD Rd, Rv, Rc1.reuse, Rd    on the FMA pipe; the scale c1 is fixed, so from the
//                                                                     reuse cache: two fresh vector reads
//   MIXM   MERGE and IMADM alternating                                one ALU-pipe and one FMA-pipe merge
//   FADDM  y[u] = f32(y[u]) + f32(y[v]) FADD Rd, Rd, Rv                two vector reads on the FP32 pipe; exact integer
//                                                                     addition for words below 2^24 (a 16-bit output)
//   MIXF   MERGE and FADDM alternating; MIX3  MERGE, IMADM, FADDM in turn: merges spread over two and three pipes
// (An immediate scale, `IMAD Rd, Rv, imm, Rd`, would read the same two registers, but ptxas 13.0 drops that loop-carried
// IMAD from the probe and filler loops and folds the values from before the loop.)
// PRMT (Round 1's `__byte_perm(y[u], y[v], imm)`, SASS `PRMT Rd, Ra, imm, Rb`) is already a two-register byte-packing
// merge, so Round 2 reuses it. UNIF and CBANK read only one vector register, but a uniform/constant addend is identical
// across a warp's 32 lanes, so the note argues what such a merge can and cannot combine.
enum Op {
  O_NONE = 0, O_IADD3, O_LOP3, O_IMAD, O_PRMT, O_MIX, O_IMMA, O_MERGE, O_REUSE, O_UNIF, O_CBANK, O_LOP3X, O_IMADM, O_MIXM,
  O_FADDM, O_MIXF, O_MIX3,
  NOPS
};
static const char* OP_NAME[NOPS] = {"none",  "iadd3", "lop3",  "imad",  "prmt", "mix",  "imma",  "merge", "reuse",
                                    "unif",  "cbank", "lop3x", "imadm", "mixm", "faddm", "mixf", "mix3"};

__device__ __forceinline__ uint32_t maj3(uint32_t a, uint32_t b, uint32_t c) { return (a & b) | (a & c) | (b & c); }

// A warp-uniform addend: derived only from %ctaid.x (the same for every lane of the warp), so ptxas keeps it in a
// uniform register and the add reads one vector register.
__device__ __forceinline__ uint32_t unif_addend() {
  uint32_t b;
  asm("mov.u32 %0, %%ctaid.x;" : "=r"(b));
  return b ^ 0x9E3779B1u;
}

// NY (a power of two) accumulators. A body of PB ops touches each accumulator at most once when PB <= NY, so no two of
// its ops fold (the probe uses NY = 16 at PB = 16; the filler's in-loop extras use NY = 8 at X <= 8). The addend of a
// merge is a computed word: MERGE adds another accumulator, REUSE one fixed computed word c[0] (served from the reuse
// cache after the first op), LOP3X/PRMT byte-merge two accumulators. UNIF and CBANK add a warp-uniform word and a
// constant: one vector read each, but identical across the warp's lanes.
template <int OP, int NY>
__device__ __forceinline__ void xop(uint32_t (&y)[NY], const uint32_t (&c)[4], int e) {
  const int u = e & (NY - 1), v = (e + 3) & (NY - 1);
  const int op = OP == O_MIX    ? ((e & 1) ? O_IMAD : O_IADD3)
                 : OP == O_MIXM ? ((e & 1) ? O_IMADM : O_MERGE)
                 : OP == O_MIXF ? ((e & 1) ? O_FADDM : O_MERGE)
                 : OP == O_MIX3 ? (e % 3 == 0 ? O_MERGE : e % 3 == 1 ? O_IMADM : O_FADDM)
                                : OP;
  if (op == O_IADD3) y[u] = y[u] + y[v] + c[e & 3];
  if (op == O_LOP3) y[u] = maj3(y[u], y[v], c[e & 3]);
  if (op == O_IMAD) y[u] = y[u] * c[e & 3] + y[v];
  if (op == O_PRMT) y[u] = __byte_perm(y[u], y[v], 0x5140);
  if (op == O_MERGE) y[u] = y[u] + y[v];
  if (op == O_IMADM) y[u] = y[v] * c[1] + y[u];
  if (op == O_FADDM) y[u] = __float_as_uint(__fadd_rn(__uint_as_float(y[u]), __uint_as_float(y[v])));
  if (op == O_REUSE) y[u] = y[u] + c[0];
  if (op == O_UNIF) y[u] = y[u] + unif_addend();
  if (op == O_CBANK) y[u] = y[u] + KK[e & (NKEY - 1)];
  if (op == O_LOP3X) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, 0x00ff00ff, 0xe4;" : "=r"(r) : "r"(y[u]), "r"(y[v]));
    y[u] = r;
  }
}

constexpr int FT = 4, OUTW = 24, MAXW = 32, NSMID = 1024;

// One loop trip: UNR rounds of FT MMAs, each followed by the four chains' absorbs, with the trip's X extras spread
// over the rounds (round q gets extras [q·X/UNR, (q+1)·X/UNR)).
template <int XOP, int X, int UNR>
__device__ __forceinline__ void trip(uint32_t (&acc)[FT][4], St (&s)[FT], const uint32_t (&a)[4],
                                     const uint32_t (&b)[2][2], uint32_t (&y)[8], const uint32_t (&c)[4]) {
#pragma unroll
  for (int q = 0; q < UNR; ++q) {
#pragma unroll
    for (int j = 0; j < FT; ++j) mma_s8(acc[j], a, b[j & 1]);
    duo4<0>(s[0], acc[0]);
    duo4<4>(s[1], acc[1]);
    duo4<8>(s[2], acc[2]);
    duo4<12>(s[3], acc[3]);
    if constexpr (X > 0) {
#pragma unroll
      for (int e = q * X / UNR; e < (q + 1) * X / UNR; ++e) xop<XOP, 8>(y, c, e);
    }
  }
}

// A probe warp: a dependent-free stream of POP in bodies of PB ops over NY accumulators, with an optional __nanosleep
// between bodies, until *vd reaches target. pinfo slot blockIdx.x · MAXW + warp: bodies, t0, t1, fold.
template <int POP, int PB, int NY>
__device__ __forceinline__ void probe(long long* pinfo, const volatile int* vd, int target, int sleep_ns, uint32_t seed,
                                      int warp, int lane) {
  {
    const uint32_t ptid = 0x100000u + blockIdx.x * 1024 + threadIdx.x;
    const int fw = target;
    long long n = 0, t0 = 0, t1 = 0;
    uint32_t fold = 0;
    if constexpr (POP == O_IMMA) {
      uint32_t pa[4], pb[2][2], pacc[8][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) pa[i] = gen(seed, ptid, 7, i);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int k = 0; k < 2; ++k) pb[i][k] = gen(seed, ptid, 8 + i, k);
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int r = 0; r < 4; ++r) pacc[e][r] = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) asm volatile("" : "+r"(pa[i]));
      asm volatile("" : "+r"(pb[0][0]), "+r"(pb[0][1]), "+r"(pb[1][0]), "+r"(pb[1][1]));
      t0 = clock64();
      while (*vd < fw) {
#pragma unroll
        for (int e = 0; e < PB; ++e) mma_s8(pacc[e & 7], pa, pb[e & 1]);
        ++n;
        if (sleep_ns > 0) __nanosleep(sleep_ns);
      }
      t1 = clock64();
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int r = 0; r < 4; ++r) fold ^= pacc[e][r];
    } else {
      uint32_t y[NY], c[4];
#pragma unroll
      for (int u = 0; u < NY; ++u) y[u] = gen(seed, ptid, 20 + u, 0);
#pragma unroll
      for (int r = 0; r < 4; ++r) c[r] = gen(seed, ptid, 30 + r, 0) | 1u;
#pragma unroll
      for (int u = 0; u < NY; ++u) asm volatile("" : "+r"(y[u]));
#pragma unroll
      for (int r = 0; r < 4; ++r) asm volatile("" : "+r"(c[r]));
      t0 = clock64();
      while (*vd < fw) {
#pragma unroll
        for (int e = 0; e < PB; ++e) xop<POP, NY>(y, c, e);
        ++n;
        if (sleep_ns > 0) __nanosleep(sleep_ns);
      }
      t1 = clock64();
#pragma unroll
      for (int u = 0; u < NY; ++u) fold ^= y[u];
    }
    if (lane == 0) {
      long long* p = pinfo + 4 * (blockIdx.x * MAXW + warp);
      p[0] = n;
      p[1] = t0;
      p[2] = t1;
      p[3] = fold;
    }
  }
}

// out: OUTW words per filler thread (4 receipts, 16 accumulator words, the extras' fold). fclk: t0, t1 per filler warp
// slot (block · MAXW + warp). pinfo: bodies, t0, t1, fold per probe warp slot. smid: per block. A probe body is PB ops.
// gdone: indexed by SM id; every filler warp adds 1 to its SM's word when it ends, and gdone[NSMID + SM id] is set when
// the SM's filler block starts (what a probe kernel on another stream waits for).
template <int POP, int XOP, int X, int UNR, int PB>
__global__ void __launch_bounds__(1024, 1)
    fk(uint32_t* out, long long* fclk, long long* pinfo, uint32_t* smid, int* gdone, int iters, int fw, int sleep_ns,
       uint32_t seed) {
  __shared__ int done;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    done = 0;
    uint32_t id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    smid[blockIdx.x] = id;
    atomicExch(gdone + NSMID + id, 1);
  }
  __syncthreads();
  if (warp < fw) {
    const uint32_t ftid = (blockIdx.x * fw + warp) * 32 + lane;
    uint32_t a[4], b[2][2], acc[FT][4], y[8], c[4];
    St s[FT];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = gen(seed, ftid, 7, i);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int k = 0; k < 2; ++k) b[i][k] = gen(seed, ftid, 8 + i, k);
#pragma unroll
    for (int j = 0; j < FT; ++j) {
      s[j].y = gen(seed, ftid, j, 100);
      s[j].z = gen(seed, ftid, j, 101);
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[j][r] = 0;
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) y[u] = gen(seed, ftid, 20 + u, 0);
#pragma unroll
    for (int r = 0; r < 4; ++r) c[r] = gen(seed, ftid, 30 + r, 0) | 1u;
    // Opaque to ptxas, so it keeps the operands in registers instead of recomputing gen() inside the timed loop.
#pragma unroll
    for (int i = 0; i < 4; ++i) asm volatile("" : "+r"(a[i]), "+r"(c[i]));
    asm volatile("" : "+r"(b[0][0]), "+r"(b[0][1]), "+r"(b[1][0]), "+r"(b[1][1]));
#pragma unroll
    for (int j = 0; j < FT; ++j) asm volatile("" : "+r"(s[j].y), "+r"(s[j].z));
#pragma unroll
    for (int u = 0; u < 8; ++u) asm volatile("" : "+r"(y[u]));
    __syncwarp();
    const long long t0 = clock64();
#pragma unroll 1
    for (int it = 0; it < iters; it += UNR) trip<XOP, X, UNR>(acc, s, a, b, y, c);
    const long long t1 = clock64();
    uint32_t* o = out + (size_t)ftid * OUTW;
#pragma unroll
    for (int j = 0; j < FT; ++j) {
      o[j] = finish(s[j]);
#pragma unroll
      for (int r = 0; r < 4; ++r) o[4 + 4 * j + r] = acc[j][r];
    }
    uint32_t fold = 0;
#pragma unroll
    for (int u = 0; u < 8; ++u) fold ^= y[u];
    o[20] = X > 0 ? fold : 0u;
    o[21] = o[22] = o[23] = 0;
    if (lane == 0) {
      fclk[2 * (blockIdx.x * MAXW + warp)] = t0;
      fclk[2 * (blockIdx.x * MAXW + warp) + 1] = t1;
    }
    __syncwarp();
    if (lane == 0) {
      uint32_t id;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
      atomicAdd(&done, 1);
      atomicAdd(gdone + id, 1);
    }
  } else if constexpr (POP != O_NONE) {
    probe<POP, PB, 8>(pinfo, &done, fw, sleep_ns, seed, warp, lane);
  }
}

// The probe as its own kernel, launched on a second stream after the filler (method a, "second stream": the filler's
// binary is the plain base-u1, untouched): PW warps a block, one block an SM beside the filler's (dynamic shared memory
// sizes, see launch), until the FW filler warps on its own SM have ended (gdone[smid] reaches FW). first: launched before
// the filler, so it first waits for its SM's filler block to start.
template <int POP, int PB>
__global__ void __launch_bounds__(1024, 1)
    pk(long long* pinfo, uint32_t* smid, const int* gdone, int fw, int sleep_ns, uint32_t seed, int first) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  if (threadIdx.x == 0) smid[blockIdx.x] = id;
  if (first)
    while (*(const volatile int*)(gdone + NSMID + id) == 0) __nanosleep(256);
  probe<POP, PB, 16>(pinfo, (const volatile int*)(gdone + id), fw, sleep_ns, seed, warp, lane);
}

// ---------------------------------------------------------------------------------------------------------------- host
using KFn = void (*)(uint32_t*, long long*, long long*, uint32_t*, int*, int, int, int, uint32_t);
using PFn = void (*)(long long*, uint32_t*, const int*, int, int, uint32_t, int);
struct Kern {
  std::string name;
  int pop, xop, x, unr, pb;
  KFn fn;
};

#define BASE(U) {"base-u" #U, O_NONE, O_NONE, 0, U, 0, fk<O_NONE, O_NONE, 0, U, 1>}
#define XK(OP, X, U) {std::string("x-") + OP_NAME[OP] + "-u" #U "-x" #X, O_NONE, OP, X, U, 0, fk<O_NONE, OP, X, U, 1>}
#define XROW1(OP) XK(OP, 1, 1), XK(OP, 2, 1), XK(OP, 3, 1), XK(OP, 4, 1), XK(OP, 6, 1), XK(OP, 8, 1), XK(OP, 12, 1), XK(OP, 16, 1)
#define XROW4(OP) XK(OP, 1, 4), XK(OP, 2, 4), XK(OP, 4, 4)
#define XROW2(OP) XK(OP, 1, 1), XK(OP, 2, 1), XK(OP, 4, 1), XK(OP, 8, 1)
#define PK(OP, PB) {std::string("p-") + OP_NAME[OP] + "-b" #PB, OP, O_NONE, 0, 1, PB, fk<OP, O_NONE, 0, 1, PB>}

// Second-stream probes (pk), beside base-u1.
struct SKern {
  std::string name;
  int pop, pb;
  PFn fn;
};
#define SK(OP, PB) {std::string("s-") + OP_NAME[OP] + "-b" #PB, OP, PB, pk<OP, PB>}
static std::vector<SKern> skernels() {
  return {SK(O_IADD3, 16), SK(O_IMAD, 16), SK(O_MIX, 16),  SK(O_IMMA, 32),  SK(O_MERGE, 16), SK(O_REUSE, 16),
          SK(O_UNIF, 16),  SK(O_CBANK, 16), SK(O_PRMT, 16), SK(O_LOP3X, 16), SK(O_IMADM, 16), SK(O_MIXM, 16),
          SK(O_FADDM, 16), SK(O_MIXF, 16),  SK(O_MIX3, 16)};
}

static std::vector<Kern> kernels() {
  return {BASE(1),          BASE(4),          XROW1(O_IADD3),   XROW1(O_LOP3),   XROW1(O_IMAD),  XROW1(O_PRMT),
          XROW1(O_MIX),     XROW4(O_IADD3),   XROW4(O_IMAD),    XROW4(O_MIX),    PK(O_IADD3, 256), PK(O_LOP3, 256),
          PK(O_IMAD, 256),  PK(O_PRMT, 256),  PK(O_MIX, 256),   PK(O_IMMA, 32),  PK(O_IADD3, 16), PK(O_IMAD, 16),
          PK(O_MIX, 16),    XROW2(O_MERGE),   XROW2(O_REUSE),   XROW2(O_UNIF),   XROW2(O_CBANK),  XROW2(O_LOP3X),
          XROW2(O_IMADM),   XROW2(O_MIXM),    XROW2(O_FADDM),   XROW2(O_MIX3)};
}

static void set_keys(uint32_t kseed) {
  uint32_t ka[NKEY], kk[NKEY];
  for (int i = 0; i < NKEY; ++i) {
    ka[i] = gen(kseed, 1, i, 0) | 1u;
    kk[i] = gen(kseed, 3, i, 0);
  }
  CK(cudaMemcpyToSymbol(KA, ka, sizeof ka));
  CK(cudaMemcpyToSymbol(KK, kk, sizeof kk));
}

// smem: what a filler block reserves alone (one block an SM). With a second-stream probe the filler reserves SS_F and
// the probe SS_P: two fillers or a filler and two probes don't fit an SM's shared memory, a filler and one probe do.
// Probe first (the covert kernel resident before the filler, so its warps are the older ones): the probe reserves SSF_P,
// so one probe block an SM, and the filler SSF_F, which fits beside it once.
constexpr int SS_F = 56 * 1024, SS_P = 36 * 1024, SSF_P = 50 * 1024, SSF_F = 47 * 1024;
struct Dev {
  int sms = 0, smem = 0;
  uint32_t *out = nullptr, *smid = nullptr, *smid2 = nullptr;
  long long *fclk = nullptr, *pinfo = nullptr;
  int* gdone = nullptr;
  cudaEvent_t e0, e1;
  cudaStream_t s1, s2;
  size_t out_words = 0;
};

static Dev make_dev() {
  Dev d;
  int dev = 0, per_smp = 0, reserved = 0, optin = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaDeviceGetAttribute(&per_smp, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  CK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev));
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  d.smem = std::min(per_smp - reserved, optin) - 256;  // the static `done` word too
  d.out_words = (size_t)d.sms * MAXW * 32 * OUTW;
  CK(cudaMalloc(&d.out, d.out_words * sizeof(uint32_t)));
  CK(cudaMalloc(&d.fclk, (size_t)d.sms * MAXW * 2 * sizeof(long long)));
  CK(cudaMalloc(&d.pinfo, (size_t)d.sms * MAXW * 4 * sizeof(long long)));
  CK(cudaMalloc(&d.smid, (size_t)d.sms * sizeof(uint32_t)));
  CK(cudaMalloc(&d.smid2, (size_t)d.sms * sizeof(uint32_t)));
  CK(cudaMalloc(&d.gdone, 2 * NSMID * sizeof(int)));
  CK(cudaEventCreate(&d.e0));
  CK(cudaEventCreate(&d.e1));
  CK(cudaStreamCreateWithFlags(&d.s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&d.s2, cudaStreamNonBlocking));
  if (d.smem < SS_F + SS_P || d.smem < SSF_F + SSF_P) {
    fprintf(stderr, "shared memory %d < %d\n", d.smem, std::max(SS_F + SS_P, SSF_F + SSF_P));
    exit(3);
  }
  for (const Kern& k : kernels()) {
    CK(cudaFuncSetAttribute(k.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, d.smem));
    CK(cudaFuncSetAttribute(k.fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  }
  for (const SKern& k : skernels()) {
    CK(cudaFuncSetAttribute(k.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, std::max(SS_P, SSF_P)));
    CK(cudaFuncSetAttribute(k.fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  }
  return d;
}

struct Result {
  double ms = 0, mhz = 0, cyc_med = 0, rate_med = 0, rate_min = 0, probe_med = 0, probe_min = 0, sleep_frac = 0;
  int distinct = 0, blocks = 0, pdistinct = 0, paired = 0;
};

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

// One launch over `blocks` blocks of FW + PW warps (FW filler warps and PW of the kernel's own probe), or, with sk, FW
// filler warps a block (shared memory SS_F) and then sk's PW probe warps a block on a second stream (SS_P; blocks must be
// every SM, so each probe block has a filler block beside it). rate: filler warp-MMAs per SM-clock over the block's
// filler span (first t0 to last t1 of its filler warps); probe: probe instructions (bodies · PB) per SM-clock over the
// span of the filler block on the same SM. smem: the filler's (0: d.smem, one block an SM).
static Result launch(Dev& d, const Kern& k, int blocks, int fw, int pw, int sleep_ns, int iters, uint32_t seed,
                     const SKern* sk = nullptr, int smem = 0, bool first = false) {
  const int threads = 32 * (fw + (sk ? 0 : pw));
  if (sk && blocks != d.sms) {
    fprintf(stderr, "a second-stream probe needs a filler block on every SM\n");
    exit(2);
  }
  CK(cudaMemset(d.out, 0xA5, d.out_words * sizeof(uint32_t)));
  CK(cudaMemset(d.fclk, 0, (size_t)d.sms * MAXW * 2 * sizeof(long long)));
  CK(cudaMemset(d.pinfo, 0, (size_t)d.sms * MAXW * 4 * sizeof(long long)));
  CK(cudaMemset(d.gdone, 0, 2 * NSMID * sizeof(int)));
  CK(cudaMemset(d.smid2, 0xFF, (size_t)d.sms * sizeof(uint32_t)));
  CK(cudaDeviceSynchronize());
  if (sk && first) sk->fn<<<blocks, 32 * pw, SSF_P, d.s2>>>(d.pinfo, d.smid2, d.gdone, fw, sleep_ns, seed, 1);
  CK(cudaEventRecord(d.e0, d.s1));
  k.fn<<<blocks, threads, sk ? (first ? SSF_F : SS_F) : (smem ? smem : d.smem), d.s1>>>(d.out, d.fclk, d.pinfo, d.smid,
                                                                                        d.gdone, iters, fw, sleep_ns, seed);
  if (sk && !first) sk->fn<<<blocks, 32 * pw, SS_P, d.s2>>>(d.pinfo, d.smid2, d.gdone, fw, sleep_ns, seed, 0);
  CK(cudaEventRecord(d.e1, d.s1));
  CK(cudaEventSynchronize(d.e1));
  CK(cudaStreamSynchronize(d.s2));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, d.e0, d.e1));
  std::vector<long long> fc((size_t)blocks * MAXW * 2), pi((size_t)blocks * MAXW * 4);
  std::vector<uint32_t> sm(blocks), sm2(blocks);
  CK(cudaMemcpy(fc.data(), d.fclk, fc.size() * sizeof(long long), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(pi.data(), d.pinfo, pi.size() * sizeof(long long), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sm.data(), d.smid, sm.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sm2.data(), d.smid2, sm2.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  // second stream: probe instructions on each SM, and how many probe blocks each SM ran
  std::vector<double> sops(NSMID, 0.0);
  std::vector<int> sblocks(NSMID, 0);
  if (sk)
    for (int bl = 0; bl < blocks; ++bl) {
      if (sm2[bl] >= (uint32_t)NSMID) continue;
      sblocks[sm2[bl]]++;
      for (int w = 0; w < pw; ++w) sops[sm2[bl]] += (double)pi[4 * (bl * MAXW + w)] * sk->pb;
    }
  Result r;
  r.ms = ms;
  r.blocks = blocks;
  std::vector<double> rate, probe, span;
  for (int bl = 0; bl < blocks; ++bl) {
    long long lo = 0, hi = 0;
    for (int w = 0; w < fw; ++w) {
      const long long t0 = fc[2 * (bl * MAXW + w)], t1 = fc[2 * (bl * MAXW + w) + 1];
      lo = w == 0 ? t0 : std::min(lo, t0);
      hi = w == 0 ? t1 : std::max(hi, t1);
    }
    const double sp = (double)(hi - lo);
    span.push_back(sp);
    rate.push_back((double)fw * iters * FT / sp);
    double ops = 0;
    if (sk) {
      ops = sm[bl] < (uint32_t)NSMID ? sops[sm[bl]] : 0;
      r.paired += sm[bl] < (uint32_t)NSMID && sblocks[sm[bl]] == 1;
    } else {
      for (int w = fw; w < fw + pw; ++w) ops += (double)pi[4 * (bl * MAXW + w)] * k.pb;
    }
    probe.push_back(ops / sp);
  }
  std::vector<uint32_t> u(sm);
  std::sort(u.begin(), u.end());
  r.distinct = (int)(std::unique(u.begin(), u.end()) - u.begin());
  if (sk) {
    std::vector<uint32_t> u2(sm2);
    std::sort(u2.begin(), u2.end());
    r.pdistinct = (int)(std::unique(u2.begin(), u2.end()) - u2.begin());
  }
  r.cyc_med = median(span);
  r.mhz = *std::max_element(span.begin(), span.end()) / (ms * 1e3);
  r.rate_med = median(rate);
  r.rate_min = *std::min_element(rate.begin(), rate.end());
  r.probe_med = median(probe);
  r.probe_min = *std::min_element(probe.begin(), probe.end());
  return r;
}

static const Kern& find(const std::vector<Kern>& ks, const std::string& name) {
  for (const Kern& k : ks)
    if (k.name == name) return k;
  fprintf(stderr, "no kernel %s\n", name.c_str());
  exit(2);
}

static void emit(const char* phase, const Kern& k, int fw, int pw, int sleep_ns, int iters, int rep, uint32_t seed,
                 const Result& r) {
  printf("{\"phase\":\"%s\",\"kernel\":\"%s\",\"pop\":\"%s\",\"pb\":%d,\"xop\":\"%s\",\"x\":%d,\"unr\":%d,\"k\":%.4f,"
         "\"fw\":%d,\"pw\":%d,\"sleep_ns\":%d,\"iters\":%d,\"rep\":%d,\"seed\":%u,\"blocks\":%d,\"distinct_sms\":%d,"
         "\"ms\":%.5f,\"cycles_med\":%.0f,\"mhz\":%.2f,\"mma_per_sm_clk_med\":%.7f,\"mma_per_sm_clk_min\":%.7f,"
         "\"probe_inst_per_sm_clk_med\":%.6f,\"probe_inst_per_sm_clk_min\":%.6f,\"probe_distinct_sms\":%d,"
         "\"paired_sms\":%d}\n",
         phase, k.name.c_str(), OP_NAME[k.pop], k.pb, OP_NAME[k.xop], k.x, k.unr, (double)k.x / (FT * k.unr), fw, pw,
         sleep_ns, iters, rep, seed, r.blocks, r.distinct, r.ms, r.cyc_med, r.mhz, r.rate_med, r.rate_min, r.probe_med,
         r.probe_min, r.pdistinct, r.paired);
  fflush(stdout);
}

// "V KERNEL SEED KSEED ITERS FTID w0..w19": sampled filler threads of a timed launch, for gate.py.
static void emit_v(const Dev& d, const Kern& k, uint32_t seed, uint32_t kseed, int iters, FILE* vf) {
  static const uint32_t tids[3] = {0, 77, 5003};
  for (uint32_t t : tids) {
    uint32_t w[OUTW];
    CK(cudaMemcpy(w, d.out + (size_t)t * OUTW, sizeof w, cudaMemcpyDeviceToHost));
    fprintf(vf, "V %s %u %u %d %u", k.name.c_str(), seed, kseed, iters, t);
    for (int i = 0; i < 20; ++i) fprintf(vf, " %08x", w[i]);
    fprintf(vf, "\n");
  }
}

#ifndef FILLER_LIB  // serve.cu includes the kernels without the benchmark
static const int SLEEPS[] = {0, 400, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 409600, 1000000};

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: filler gate|sweep|one|list ...\n");
    return 2;
  }
  const std::string mode = argv[1];
  const std::vector<Kern> ks = kernels();
  if (mode == "list") {
    for (const Kern& k : ks) printf("%s\n", k.name.c_str());
    for (const SKern& k : skernels()) printf("%s\n", k.name.c_str());
    return 0;
  }
  const uint32_t kseed0 = 0xC0FFEE11u;
  Dev d = make_dev();
  if (mode == "gate") {
    const uint32_t seed = argc > 2 ? (uint32_t)strtoul(argv[2], 0, 0) : 0x5EED1234u;
    const uint32_t kseed = argc > 3 ? (uint32_t)strtoul(argv[3], 0, 0) : kseed0;
    set_keys(kseed);
    const int fw = 4, pw = 4, blocks = 2;
    std::vector<uint32_t> h((size_t)blocks * fw * 32 * OUTW);
    for (const Kern& k : ks)
      for (int iters : {4, 12}) {
        launch(d, k, blocks, fw, k.pop ? pw : 0, 0, iters, seed);
        CK(cudaMemcpy(h.data(), d.out, h.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int t = 0; t < blocks * fw * 32; ++t) {
          printf("G %s %u %u %d %d", k.name.c_str(), seed, kseed, iters, t);
          for (int i = 0; i < 20; ++i) printf(" %08x", h[(size_t)t * OUTW + i]);
          printf("\n");
        }
      }
    // Second-stream probes beside base-u1 (a filler block on every SM; the first two blocks' threads printed).
    const Kern& b1 = find(ks, "base-u1");
    for (const SKern& p : skernels())
      for (int iters : {4, 12, -4, -12}) {
        launch(d, b1, d.sms, fw, pw, 0, std::abs(iters), seed, &p, 0, iters < 0);
        CK(cudaMemcpy(h.data(), d.out, h.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int t = 0; t < blocks * fw * 32; ++t) {
          printf("G ss%s-%s %u %u %d %d", iters < 0 ? "f" : "", p.name.c_str(), seed, kseed, std::abs(iters), t);
          for (int i = 0; i < 20; ++i) printf(" %08x", h[(size_t)t * OUTW + i]);
          printf("\n");
        }
      }
    return 0;
  }
  if (mode == "ssdump") {  // ssdump NAME FW PW SLEEP ITERS [FIRST]: per SM, the filler block's span and the probe's warps
    if (argc < 7) return 2;
    set_keys(kseed0);
    const int fw = atoi(argv[3]), pw = atoi(argv[4]);
    for (const SKern& p : skernels())
      if (p.name == argv[2]) {
        launch(d, find(ks, "base-u1"), d.sms, fw, pw, atoi(argv[5]), atoi(argv[6]), 0x5EED1234u, &p, 0,
               argc > 7 && atoi(argv[7]));
        std::vector<long long> fc((size_t)d.sms * MAXW * 2), pi((size_t)d.sms * MAXW * 4);
        std::vector<uint32_t> sm(d.sms), sm2(d.sms);
        CK(cudaMemcpy(fc.data(), d.fclk, fc.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pi.data(), d.pinfo, pi.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(sm.data(), d.smid, sm.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(sm2.data(), d.smid2, sm2.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int bl = 0; bl < 6; ++bl) {
          long long lo = fc[2 * bl * MAXW], hi = fc[2 * bl * MAXW + 1];
          for (int w = 1; w < fw; ++w) {
            lo = std::min(lo, fc[2 * (bl * MAXW + w)]);
            hi = std::max(hi, fc[2 * (bl * MAXW + w) + 1]);
          }
          for (int pb = 0; pb < d.sms; ++pb)
            if (sm2[pb] == sm[bl])
              for (int w = 0; w < pw; ++w) {
                const long long* q = &pi[4 * (pb * MAXW + w)];
                printf("D sm %u filler %lld..%lld (%lld) probe-block %d warp %d bodies %lld t0-lo %lld t1-hi %lld\n", sm[bl], lo,
                       hi, hi - lo, pb, w, q[0], q[1] - lo, q[2] - hi);
              }
        }
        return 0;
      }
    return 2;
  }
  if (mode == "one") {
    if (argc < 7) return 2;
    set_keys(kseed0);
    const std::string name = argv[2];
    for (const SKern& p : skernels())
      if (p.name == name) {
        launch(d, find(ks, "base-u1"), d.sms, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), 0x5EED1234u, &p);
        return 0;
      }
    const Kern& k = find(ks, name);
    launch(d, k, d.sms, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), 0x5EED1234u);
    return 0;
  }
  if (mode != "sweep") return 2;
  const int iters = (argc > 2 ? atoi(argv[2]) : 32000) & ~3, reps = argc > 3 ? atoi(argv[3]) : 7;
  FILE* vf = fopen(argc > 4 ? argv[4] : "verify.txt", "w");
  set_keys(kseed0);
  const Kern& b1 = find(ks, "base-u1");
  const Kern& b4 = find(ks, "base-u4");
  // Warm up.
  launch(d, b1, d.sms, 16, 0, 0, iters, 1);
  for (int rep = 0; rep < reps; ++rep) {
    const uint32_t seed = 0x5EED0000u + rep;
    // Occupancy: the filler alone at 1..8 warps per SMSP.
    for (int fw : {4, 8, 12, 16, 20, 24, 28, 32}) {
      Result r = launch(d, b1, d.sms, fw, 0, 0, iters, seed);
      emit("occupancy", b1, fw, 0, 0, iters, rep, seed, r);
      if (fw == 16) emit_v(d, b1, seed, kseed0, iters, vf);
    }
    for (int fw : {8, 16}) emit("occupancy", b4, fw, 0, 0, iters, rep, seed, launch(d, b4, d.sms, fw, 0, 0, iters, seed));
    // Method b: extras in the filler's loop, each beside its own base, at 2 and 4 filler warps per SMSP.
    for (int fw : {8, 16})
      for (const Kern& k : ks) {
        if (k.xop == O_NONE) continue;
        const Kern& b = k.unr == 1 ? b1 : b4;
        emit("inloop-base", b, fw, 0, 0, iters, rep, seed, launch(d, b, d.sms, fw, 0, 0, iters, seed));
        Result r = launch(d, k, d.sms, fw, 0, 0, iters, seed);
        emit("inloop", k, fw, 0, 0, iters, rep, seed, r);
        if (fw == 16) emit_v(d, k, seed, kseed0, iters, vf);
      }
    // Method a: probe warps beside the filler at its full-rate occupancy (4 filler warps per SMSP). Each probe launch
    // follows the same kernel with no probe warps (the same filler code: ptxas compiles the filler loop differently once
    // a probe path is in the kernel), and each kernel's group starts with the plain filler (base-u1).
    for (const Kern& k : ks) {
      if (k.pop == O_NONE) continue;
      const int fw = 16;
      emit("probe-plain", b1, fw, 0, 0, iters, rep, seed, launch(d, b1, d.sms, fw, 0, 0, iters, seed));
      for (int pw : {4, 8, 16})
        for (int sl : SLEEPS) {
          emit("probe-base", k, fw, 0, 0, iters, rep, seed, launch(d, k, d.sms, fw, 0, 0, iters, seed));
          Result r = launch(d, k, d.sms, fw, pw, sl, iters, seed);
          emit("probe", k, fw, pw, sl, iters, rep, seed, r);
          if (pw == 4 && sl == 0) emit_v(d, k, seed, kseed0, iters, vf);
        }
    }
    // Method a, second stream: the plain filler (base-u1's own binary, shared memory SS_F) and a probe kernel on a
    // second stream, one probe block beside each filler block; each launch follows the filler alone at SS_F.
    for (const SKern& p : skernels()) {
      const int fw = 16;
      const Kern pkn{p.name, p.pop, O_NONE, 0, 1, p.pb, b1.fn};
      for (int pw : {4, 8, 16})
        for (int sl : SLEEPS) {
          emit("ss-base", b1, fw, 0, 0, iters, rep, seed, launch(d, b1, d.sms, fw, 0, 0, iters, seed, nullptr, SS_F));
          Result r = launch(d, b1, d.sms, fw, pw, sl, iters, seed, &p);
          emit("ss-probe", pkn, fw, pw, sl, iters, rep, seed, r);
          if (pw == 4 && sl == 0) emit_v(d, pkn, seed, kseed0, iters, vf);
        }
    }
    // The same with the probe kernel resident first (its warps the older ones), for the IMMA and IADD3 probes and the
    // Round 2 merge-capable forms.
    for (const SKern& p : skernels()) {
      if (p.pop != O_IMMA && p.pop != O_IADD3 && p.pop != O_PRMT && p.pop < O_MERGE) continue;
      const int fw = 16;
      const Kern pkn{p.name, p.pop, O_NONE, 0, 1, p.pb, b1.fn};
      for (int pw : {4, 8, 16})
        for (int sl : SLEEPS) {
          emit("ss-base", b1, fw, 0, 0, iters, rep, seed, launch(d, b1, d.sms, fw, 0, 0, iters, seed, nullptr, SS_F));
          Result r = launch(d, b1, d.sms, fw, pw, sl, iters, seed, &p, 0, true);
          emit("ssf-probe", pkn, fw, pw, sl, iters, rep, seed, r);
          if (pw == 4 && sl == 0) emit_v(d, pkn, seed, kseed0, iters, vf);
        }
    }
  }
  fclose(vf);
  return 0;
}
#endif
