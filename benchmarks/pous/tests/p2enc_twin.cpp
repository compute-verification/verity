// A host build of p2_v1/p2enc.cuh's encode kernel, lane for lane, for byte tests without a GPU (test_pous_p2_encode_sq.py).
//
// A warp's 32 lanes are 32 contexts on one thread. A lane runs until its next warp-wide instruction (a shuffle, a vote,
// __syncwarp), posts its operand and yields; a round resumes every lane once, in an order drawn afresh, so a lane reads
// what the others posted at the same instruction, and a read of shared memory another lane writes with no __syncwarp
// between gives a wrong value in some orders. Lanes found at different instructions abort the run. Warps run one after
// another.
//
// p2sq.cuh's squaring is PTX, so the kernel's sqr_k is replaced by the product of whole numbers (sqr_k_twin), which also
// overwrites the shared memory sqr_k uses: the test is of the encode around the squaring, not of the squaring.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ucontext.h>
#include <utility>

#define __device__
#define __global__ [[maybe_unused]] static
#define __launch_bounds__(...)
#define __forceinline__ inline
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define __grid_constant__

struct uint4 { uint32_t x, y, z, w; };
static inline uint4 make_uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return {a, b, c, d}; }
static inline uint32_t __umulhi(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * b) >> 32); }
static inline uint32_t __funnelshift_l(uint32_t lo, uint32_t hi, uint32_t n) { n &= 31; return n ? hi << n | lo >> (32 - n) : hi; }
static inline uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t s) {
  const uint32_t b[8] = {x & 255, x >> 8 & 255, x >> 16 & 255, x >> 24, y & 255, y >> 8 & 255, y >> 16 & 255, y >> 24};
  return b[s & 7] | b[s >> 4 & 7] << 8 | b[s >> 8 & 7] << 16 | b[s >> 12 & 7] << 24;
}
static inline size_t __cvta_generic_to_shared(const void *p) { return (size_t)p; }
static inline uint32_t atomicOr(uint32_t *p, uint32_t v) { const uint32_t was = *p; *p |= v; return was; }
struct Dim { unsigned x = 0, y = 0, z = 0; };
static Dim threadIdx, blockIdx, gridDim;

namespace twin {
constexpr int L = 32;
constexpr size_t STACK = 1 << 18;
enum { SYNC, SHFL, SHFL_XOR, SHFL_UP, SHFL_DOWN, BALLOT };
static ucontext_t sched, ctx[L];
static bool done[L];
static int cur;
static uint64_t seq[L];
static uint32_t post[2][L];
static uint64_t tag[2][L];
static uint32_t rng = 0x9E3779B9u;
static void (*body)();

// this lane's next warp-wide instruction: what every lane posted at it
static const uint32_t *sync(int kind, uint32_t v) {
  const int lane = cur, s = (int)(seq[lane] & 1);
  post[s][lane] = v;
  tag[s][lane] = seq[lane]++ << 3 | (uint64_t)kind;
  swapcontext(&ctx[lane], &sched);
  for (int i = 0; i < L; i++)
    if (tag[s][i] != tag[s][lane]) {                           // a lane that ended, or one elsewhere, left another tag
      fprintf(stderr, "twin: lanes %d and %d are not at the same warp-wide instruction\n", lane, i);
      abort();
    }
  return post[s];
}
static void entry() {
  body();
  done[cur] = true;
}
static void run_warp(unsigned first_thread, void (*fn)()) {
  static char *stacks = (char *)malloc(L * STACK);
  int order[L];
  body = fn;
  for (int i = 0; i < L; i++) {
    getcontext(&ctx[i]);
    ctx[i].uc_stack.ss_sp = stacks + i * STACK;
    ctx[i].uc_stack.ss_size = STACK;
    ctx[i].uc_link = &sched;
    makecontext(&ctx[i], entry, 0);
    done[i] = false;
    seq[i] = 0;
    tag[0][i] = tag[1][i] = ~(uint64_t)0;
    order[i] = i;
  }
  for (int live = L; live;) {
    for (int i = L - 1; i > 0; i--) {
      rng = rng * 1664525u + 1013904223u;
      std::swap(order[i], order[(rng >> 8) % (unsigned)(i + 1)]);
    }
    for (int o = 0; o < L; o++) {
      const int i = order[o];
      if (done[i]) continue;
      cur = i;
      threadIdx.x = first_thread + (unsigned)i;
      swapcontext(&sched, &ctx[i]);
      if (done[i]) live--;
    }
    if (live && live != L) {
      fprintf(stderr, "twin: %d lanes ended before the others\n", L - live);
      abort();
    }
  }
}
}  // namespace twin

static inline void __syncwarp(unsigned = 0xFFFFFFFFu) { twin::sync(twin::SYNC, 0); }
static inline uint32_t __shfl_sync(unsigned, uint32_t v, int src) { return twin::sync(twin::SHFL, v)[src & 31]; }
static inline uint32_t __shfl_xor_sync(unsigned, uint32_t v, int d) {
  const int me = twin::cur;
  return twin::sync(twin::SHFL_XOR, v)[me ^ d];
}
static inline uint32_t __shfl_up_sync(unsigned, uint32_t v, int d) {
  const int me = twin::cur;
  const uint32_t *p = twin::sync(twin::SHFL_UP, v);
  return me >= d ? p[me - d] : v;
}
static inline uint32_t __shfl_down_sync(unsigned, uint32_t v, int d) {
  const int me = twin::cur;
  const uint32_t *p = twin::sync(twin::SHFL_DOWN, v);
  return me + d < twin::L ? p[me + d] : v;
}
static inline uint32_t __ballot_sync(unsigned, bool b) {
  const uint32_t *p = twin::sync(twin::BALLOT, b);
  uint32_t r = 0;
  for (int i = 0; i < twin::L; i++) r |= (p[i] & 1) << i;
  return r;
}
static inline bool __all_sync(unsigned m, bool b) { return __ballot_sync(m, b) == 0xFFFFFFFFu; }
static inline bool __any_sync(unsigned m, bool b) { return __ballot_sync(m, b) != 0; }

#include "p2sq.cuh"

namespace p2sq {
static uint32_t twin_in[N], twin_out[N];
static bool twin_squared;
static uint64_t twin_squarings;

// twin_out = twin_in^2 mod p = 2^W - c0, below p; p minus that when odd is set and it isn't zero (sqr_k's contract)
static void twin_square(uint32_t c0, bool odd) {
  typedef unsigned __int128 u128;
  constexpr int H = N / 2;
  static uint64_t a[H], w[2 * H], r[H];
  for (int i = 0; i < H; i++) a[i] = twin_in[2 * i] | (uint64_t)twin_in[2 * i + 1] << 32;
  memset(w, 0, sizeof w);
  for (int i = 0; i < H; i++) {
    uint64_t carry = 0;
    for (int j = 0; j < H; j++) {
      const u128 t = (u128)a[i] * a[j] + w[i + j] + carry;
      w[i + j] = (uint64_t)t;
      carry = (uint64_t)(t >> 64);
    }
    w[i + H] = carry;
  }
  u128 c = 0;                                                  // 2^W = c0 mod p
  for (int i = 0; i < H; i++) {
    c += (u128)w[i] + (u128)w[H + i] * c0;
    r[i] = (uint64_t)c;
    c >>= 64;
  }
  for (uint64_t top = (uint64_t)c; top;) {
    u128 t = (u128)top * c0;
    for (int i = 0; i < H; i++) {
      t += r[i];
      r[i] = (uint64_t)t;
      t >>= 64;
    }
    top = (uint64_t)t;
  }
  bool ge = r[0] >= (uint64_t)0 - c0, zero = r[0] == 0;
  for (int i = 1; i < H; i++) { ge = ge && r[i] == ~(uint64_t)0; zero = zero && r[i] == 0; }
  if (ge) {
    r[0] += c0;
    for (int i = 1; i < H; i++) r[i] = 0;
    zero = r[0] == 0;
  }
  if (odd && !zero) {                                          // p - r = (2^W - 1 - r) - (c0 - 1)
    uint64_t borrow = c0 - 1;
    for (int i = 0; i < H; i++) {
      const uint64_t v = ~r[i];
      r[i] = v - borrow;
      borrow = v < borrow;
    }
  }
  for (int i = 0; i < H; i++) { twin_out[2 * i] = (uint32_t)r[i]; twin_out[2 * i + 1] = (uint32_t)(r[i] >> 32); }
}

inline void sqr_k_twin(uint32_t *sm, Num &x, bool odd, uint32_t c0, uint32_t lane) {
  for (int q = 0; q < 16; q++) twin_in[16 * lane + q] = x.l[q];
  if (lane == 31) { twin_in[512] = x.e0; twin_in[513] = x.e1; }
  __syncwarp();
  if (!twin_squared) {
    twin_square(c0, odd);
    for (int i = 0; i < KSNAP + 32 * KSNAP_STRIDE; i++) sm[i] = 0xA5000000u ^ (uint32_t)twin_squarings ^ (uint32_t)i << 12;
    twin_squarings++;
    twin_squared = true;
  }
  for (int q = 0; q < 16; q++) x.l[q] = twin_out[16 * lane + q];
  x.e0 = lane == 31 ? twin_out[512] : 0;
  x.e1 = lane == 31 ? twin_out[513] : 0;
  __syncwarp();
  twin_squared = false;
}
}  // namespace p2sq

#define sqr_k sqr_k_twin
#include "p2enc.cuh"

static struct {
  p2sq::Chain chain;
  const uint32_t *m, *k, *t;
  uint32_t *c, *flags, n, c0;
} args;

static void kernel() { encode_sq_kernel(args.chain, args.m, args.c, args.k, args.t, args.n, args.c0, args.flags); }

// p2v1_encode_sq's arguments on one thread block; squarings gets how many the kernel ran. -1 when the chain doesn't fit
extern "C" int twin_encode(const uint32_t *m, uint32_t *c, const uint32_t *k, const uint32_t *t, uint32_t n, uint32_t c0,
                           uint32_t ea, uint32_t en, uint32_t er, uint32_t *flags, uint64_t *squarings) {
  if (!p2sq::chain_for(ea, en, er, args.chain)) return -1;
  args.m = m; args.c = c; args.k = k; args.t = t; args.n = n; args.c0 = c0; args.flags = flags;
  p2sq::twin_squarings = 0;
  gridDim.x = 1;
  blockIdx.x = 0;
  for (unsigned warp = 0; warp < P2SQ_WARPS; warp++) twin::run_warp(32 * warp, kernel);
  *squarings = p2sq::twin_squarings;
  return 0;
}

// chain_for's steps into steps (CHAIN_MAX words): how many, or -1
extern "C" int twin_chain(uint32_t a, uint32_t n, uint32_t r, uint32_t *steps) {
  p2sq::Chain ch;
  if (!p2sq::chain_for(a, n, r, ch)) return -1;
  memcpy(steps, ch.step, ch.len * sizeof(uint32_t));
  return (int)ch.len;
}
extern "C" int twin_chain_max() { return (int)p2sq::CHAIN_MAX; }
