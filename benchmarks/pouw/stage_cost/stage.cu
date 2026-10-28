// What RS-D1E's stage coins and deferred edit cost on one RTX PRO 6000 (sm_120), at Llama-3.1-8B's layer shapes, int8
// (STAGE-COST, cmp-13; note:20261016T0750Z-draft-rs-d1e-stage-coins).
//
// The GEMMs are dcap.cu's: cuBLASLt int8 (s8 x s8 -> s32), the fastest of the heuristic's top 16 among those whose output
// equals the host's on sampled rows. The commitment: a stage's operand rows (k int8 bytes a token) as hm96-sha512/row-seg/v1
// leaves of a frame-v3-sha512 tree, its constants written by prep.py, by one of two paths that the gate holds to the same
// bytes: served_zk's hm96_rows.cu as is (HASH=sz), or this file's fast path (the default, fr_salt and fr_tree below: the
// same segment digests, then one launch for the leaves and the whole tree, its salt-only part computed ahead). The decode
// step beside the deferred pass is dcap.cu's (enqueue_step: 32 layers of linears and decode attention, B sequences of
// context L).
//
// Modes (one JSON line each; clocks are read, never set):
//   gate DIR            every commitment file in DIR committed into 0xA5-poisoned buffers by both paths equals Python's leaves,
//                       b || c and root, the two trees are equal node by node, and a no-write control doesn't; cuBLASLt's
//                       int8 against the host (dcap's gate)
//   prefill DIR M       one chunk's forward (32 layers, 6 stages a layer) at M tokens, by arm:
//                         gemm     the stages' GEMMs alone
//                         hash     the stages' operand commitments alone (salt, tree: the fast path's two parts alone)
//                         serial   each stage's commitment, then its GEMM, on one stream
//                         conc     the GEMMs and the commitments on two streams with no dependency (the throughput cost
//                                  of sharing the GPU, with the dependency fully hidden); conc_lo with the hashes at low
//                                  priority
//                         green N  the commitments on a green-context partition of N SMs, the GEMMs on the rest
//                         pipe C   C chunks in flight, each on its own stream (chunk 0 highest priority): for every stage,
//                                  commit its operands, wait DRAW_US for the draw, then the GEMM; C chunks' makespan
//                                  against C x gemm. pipe1_draw: the draws without commitments; pipe1_hash: the
//                                  commitments without draws
//                         pipe_green C chunks in flight on a split: each stage's commitment and draw on an N-SM partition,
//                                  its GEMM on the rest, ordered by events (host-launched), against C x gemm and against
//                                  the C chunks' GEMMs alone on the split
//                       Attention's QK^T and P.V are int8 GEMMs with the causal stage's MACs (m = M, n = M/2, k = 4096);
//                       P.V's commitment is the per-row normalizers (256 bytes a token), since P is derived.
//                       ARMS=r2: the proved forward, lockstep (commit, draw, GEMM a stage), by commitment latency lever
//                       (CFGS: a comma-separated subset by name):
//                         mid      the frame heads' first blocks compressed ahead (all arms but round 1's before)
//                         pair     the tree's levels in shared memory by warp pairs (rounds and schedule apart)
//                         s<k>     every GEMM in k row slabs, the next stage's operands hashed behind each slab on a
//                                  stream a slab; gemm_s<k> the slabbed GEMMs alone
//                         c<k>     every GEMM in k column slabs, each slab's blocks of the next stage's rows hashed behind
//                                  it (an emulated epilogue: what is left at the GEMM's end is the last slab's blocks, the
//                                  leaves and the tree); gemm_c<k> alone
//                         cpu<E>[p<P>] the tree stops at E nodes in pinned host memory, P host threads hash the top (a
//                                  subtree each) and one waits the draw, the GEMM's stream waits on its mapped flag; _nd:
//                                  no draw; _nopdl: fr_tree after its segment pass completes (every other arm launches it
//                                  programmatically, g_pdl)
//                       and the phase probe: one commitment's clocks a CTA (fr_tp)
//   shalat              a compression's latency in a dependent chain on the GPU by rotation choice (FR_ROT's bits) and split
//                       across two warps (rounds, schedule); the host's (C++, libcrypto's) and the mapped round trip
//   deferred T COMP     the noised recompute of one layer's products batched over T tokens: QKV, o, gate_up, down (in row
//                       chunks of at most 8192), and per sequence QK^T twice (clean, to rebuild P; noised) and P.V, batched
//                       over the 8 KV heads with the 4 query heads of a group stacked. COMP seq500: T/512 sequences of
//                       512 tokens; b32: 32 sequences of T/32 tokens. Contexts spread over 1K..8K; keys = context + half the
//                       sub-round's tokens (the causal mean). Arms gemm (no forming) and form (every operand's noised copy
//                       formed by a pass over it, and S -> P quantized, before the GEMMs: an unfused upper bound). Attention
//                       by ATT: fused (the default, att_fused below: S and P never leave registers, the keys split across
//                       CTAs to fill the GPU) or lt (cuBLASLt with S and P in memory); att_lt and att_fused time each alone.
//   beside T COMP B L   the deferred pass's layers kept queued beside the decode step: the step on the whole GPU, the
//                       step with the pass on a low-priority stream, the step on a green partition with the pass on
//                       FILL_SMS SMs, the step on the whole GPU again; the step's slowdown and the pass's rate in the window
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a stage.cu -lcublasLt -lnvidia-ml -ldl
#define main dcap_main
#include "../filler/dcap.cu"
#undef main
#include "../served_zk/hm96_rows.cu"

#include <immintrin.h>

#include <fstream>
#include <map>
#include <tuple>

static const char* RUN_DRAW_US = "DRAW_US";

// ---------------------------------------------------------------- the commitment (hm96_rows.cu, prep.py's constants)
static uint8_t *g_tables, *g_hm_mid;
static std::vector<uint8_t> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static uint8_t* dput(const uint8_t* h, size_t n) {
  uint8_t* p;
  CK(cudaMalloc(&p, std::max<size_t>(n, 64)));
  if (n) CK(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice));
  return p;
}
static void hm_load(const std::string& dir) {
  const std::vector<uint8_t> b = read_file(dir + "/hm.bin");
  const size_t tb = (size_t)192 * 256 * 64;
  if (b.size() != tb + 256) { fprintf(stderr, "hm.bin: %zu bytes\n", b.size()); exit(2); }
  g_tables = dput(b.data(), tb);
  uint8_t* pre = dput(b.data() + tb, 256);
  CK(cudaMalloc(&g_hm_mid, 128));
  Args a{};
  a.p[0] = (unsigned long long)pre;
  a.p[1] = (unsigned long long)g_hm_mid;
  a.i[0] = 2;
  sz_mid<<<1, 32>>>(a);
  CK(cudaDeviceSynchronize());
}

// a constants file: the prefix blocks, heads, pads, the gate's rows, salts and Python's references, and the fast path's
// padded frame templates with their values' byte offsets
struct HostConst {
  int k, R, s, last, width, nblk;
  std::vector<uint8_t> blocks, leaf_heads, node_heads, pads, rows, salts, hl, bc, root, ltpl, loff, ntpl, noff;
};
static HostConst load_const(const std::string& dir, int k, int R) {
  const std::vector<uint8_t> b = read_file(dir + "/c_" + std::to_string(k) + "_" + std::to_string(R) + ".bin");
  uint64_t h[8];
  memcpy(h, b.data(), 64);
  if (h[0] != 0x5354414745434F53ull || (int)h[1] != k || (int)h[2] != R) { fprintf(stderr, "bad constants k %d R %d\n", k, R); exit(2); }
  HostConst c;
  c.k = k; c.R = R; c.s = (int)h[3]; c.last = (int)h[4]; c.width = (int)h[5]; c.nblk = (int)h[6];
  size_t o = 64;
  auto take = [&](std::vector<uint8_t>& v, size_t n) {
    if (o + n > b.size()) { fprintf(stderr, "short constants file k %d R %d\n", k, R); exit(2); }
    v.assign(b.begin() + o, b.begin() + o + n);
    o += n;
  };
  take(c.blocks, (size_t)128 * c.nblk);
  take(c.leaf_heads, (size_t)SLOT * R);
  take(c.node_heads, (size_t)SLOT * std::max(c.width - 1, 1));
  take(c.pads, (size_t)64 * (c.width - R));
  take(c.rows, (size_t)R * k);
  take(c.salts, (size_t)192 * R);
  take(c.hl, (size_t)64 * R);
  take(c.bc, (size_t)128 * R);
  take(c.root, 64);
  const size_t nodes = (size_t)std::max(c.width - 1, 1);
  take(c.ltpl, (size_t)256 * R);
  take(c.loff, (size_t)4 * R);
  take(c.ntpl, 384 * nodes);
  take(c.noff, 4 * nodes);
  if (o != b.size()) { fprintf(stderr, "constants file k %d R %d: %zu trailing bytes\n", k, R, b.size() - o); exit(2); }
  return c;
}

// one commitment's device state: its own outputs, so commitments in flight at once never share a buffer
struct Commit {
  const HostConst* hc;
  int k, R, s, last, width;
  uint8_t *mids, *leaf_heads, *node_heads, *digests, *x, *bc, *tree, *salts;
  size_t tree_bytes;
  uint8_t *my, *cs;                    // the fast path's salt-only parts, a slot a layer (R x 64 each)
  const uint8_t *ltpl, *loff, *ntpl, *noff;
  int* counter;                        // CTAs done, for the last one to take the levels above the CTAs' subtrees
  uint8_t* fmid;                       // the frame leaves' and nodes' first-block states (R, then width - 1), computed ahead
};
struct FrConst { uint8_t *ltpl, *loff, *ntpl, *noff; };
static std::map<std::pair<int, int>, FrConst> g_fr;  // one copy of a shape's templates for all its commitments
static Commit make_commit(const HostConst& c) {
  Commit m;
  m.hc = &c; m.k = c.k; m.R = c.R; m.s = c.s; m.last = c.last; m.width = c.width;
  const std::pair<int, int> key(c.k, c.R);
  if (!g_fr.count(key))
    g_fr[key] = {dput(c.ltpl.data(), c.ltpl.size()), dput(c.loff.data(), c.loff.size()), dput(c.ntpl.data(), c.ntpl.size()),
                 dput(c.noff.data(), c.noff.size())};
  if (c.s < 1 || c.width > 128 * 1024) { fprintf(stderr, "fr_tree takes s >= 1 and at most 1024 CTAs of 128 leaves\n"); exit(2); }
  for (size_t j = 0; j < c.loff.size() + c.noff.size(); j += 4) {
    uint32_t o;
    memcpy(&o, j < c.loff.size() ? &c.loff[j] : &c.noff[j - c.loff.size()], 4);
    if (o < 128 && c.width > 1) { fprintf(stderr, "a frame head shorter than a block (%u bytes): no first-block state\n", o); exit(2); }
  }
  const FrConst& f = g_fr[key];
  m.ltpl = f.ltpl; m.loff = f.loff; m.ntpl = f.ntpl; m.noff = f.noff;
  CK(cudaMalloc(&m.fmid, (size_t)64 * (c.R + c.width - 1)));
  CK(cudaMalloc(&m.my, (size_t)64 * c.R * LAYERS));
  CK(cudaMalloc(&m.cs, (size_t)64 * c.R * LAYERS));
  CK(cudaMalloc(&m.counter, sizeof(int)));
  CK(cudaMemset(m.counter, 0, sizeof(int)));
  uint8_t* blk = dput(c.blocks.data(), c.blocks.size());
  CK(cudaMalloc(&m.mids, (size_t)64 * c.nblk));
  Args a{};
  a.p[0] = (unsigned long long)blk;
  a.p[1] = (unsigned long long)m.mids;
  a.i[0] = c.nblk;
  sz_mid<<<(c.nblk + 31) / 32, 32>>>(a);
  m.leaf_heads = dput(c.leaf_heads.data(), c.leaf_heads.size());
  m.node_heads = dput(c.node_heads.data(), c.node_heads.size());
  m.salts = dput(c.salts.data(), c.salts.size());
  CK(cudaMalloc(&m.digests, (size_t)64 * c.R * std::max(c.s, 1)));
  CK(cudaMalloc(&m.x, (size_t)64 * c.R));
  CK(cudaMalloc(&m.bc, (size_t)128 * c.R));
  m.tree_bytes = (size_t)64 * (2 * c.width - 1);
  CK(cudaMalloc(&m.tree, m.tree_bytes));
  if (!c.pads.empty()) CK(cudaMemcpy(m.tree + (size_t)64 * c.R, c.pads.data(), c.pads.size(), cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());
  CK(cudaFree(blk));
  return m;
}
// rows_bench.Commit.run(2, one_cta=False): segment digests, then x, b || c, the hm96 leaf and the frame leaf a thread a
// row, then the tree a level a launch
static void commit_run(const Commit& m, const uint8_t* data, cudaStream_t st) {
  if (m.s) {
    Args a{};
    a.p[0] = (unsigned long long)data;
    a.p[1] = (unsigned long long)m.mids;
    a.p[2] = (unsigned long long)m.digests;
    a.i[0] = m.R; a.i[1] = m.k; a.i[2] = m.s; a.i[3] = m.last;
    const long long total = (long long)m.R * m.s;
    sz_segments<<<(unsigned)((total + 255) / 256), 256, 8 * 32 * 33 * 4, st>>>(a);
  }
  Args f{};
  const unsigned long long ps[10] = {(unsigned long long)(m.s ? m.digests : data), (unsigned long long)(m.mids + 64 * m.s),
                                     (unsigned long long)g_tables, (unsigned long long)m.salts, (unsigned long long)g_hm_mid,
                                     (unsigned long long)m.leaf_heads, (unsigned long long)m.node_heads,
                                     (unsigned long long)m.x, (unsigned long long)m.bc, (unsigned long long)m.tree};
  for (int j = 0; j < 10; j++) f.p[j] = ps[j];
  f.i[0] = m.R; f.i[1] = m.s; f.i[2] = m.k; f.i[3] = 2; f.i[6] = m.width; f.i[7] = m.k;
  sz_finish<<<(unsigned)((m.R + 127) / 128), 128, 0, st>>>(f);
  long long w = m.width, off = 0;
  const uint8_t* heads = m.node_heads;
  while (w > 1) {
    Args l{};
    l.p[0] = (unsigned long long)m.tree;
    l.p[1] = (unsigned long long)heads;
    l.i[0] = (int)off; l.i[1] = (int)(w / 2);
    sz_level<<<(unsigned)((w / 2 + 127) / 128), 128, 0, st>>>(l);
    off += w;
    heads += (size_t)SLOT * (w / 2);
    w /= 2;
  }
}
// ---------------------------------------------------------------- the fast path: the same leaves, b || c, nodes and root
//   fr_salt  a thread a row, before the row exists: M_key.y and c = SHA-512(salt_prefix || y) (2 compressions, 192 table
//            entries), into a layer's slot
//   fr_tree  a thread a row: x from its segment digests, b = x ^ M_key.y, the hm96 leaf, and the frame leaf from its padded
//            template with the value inserted as words (no byte assembly); then the CTA's 128-leaf subtree in shared memory,
//            and the last CTA to finish (a counter) the levels above. One launch after fr_segments (sz_segments with
//            fr_compress).
// Rotations on the FMA pipe, which the compression otherwise leaves idle (every SHF, LOP3 and IADD costs one warp 2 cycles of
// the integer pipe): with m = 2^(32-n), x >> n = mul.hi(x, m) and (y << (32-n)) | (x >> n) = mad.lo(y, m, x >> n), the two
// parts sharing no bit. m is read from constant memory, so ptxas can't turn the multiply back into a shift. ROT's bits pick
// the functions so rotated: 1 S0, 2 S1, 4 s0, 8 s1 (FR_ROT for the commitment's kernels; shalat times each). Bits 16 and
// 32 reassociate the round's sums (fr_round), the same digests.
#ifndef FR_ROT
#define FR_ROT 0
#endif
__constant__ uint32_t FR_POW2[33] = {1u,       2u,       4u,       8u,       16u,      32u,      64u,      128u,     256u,
                                     512u,     1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15, 1u << 16, 1u << 17,
                                     1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22, 1u << 23, 1u << 24, 1u << 25, 1u << 26,
                                     1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31, 0u};
__device__ __forceinline__ uint32_t fr_mulhi(uint32_t x, uint32_t m) {
  uint32_t r;
  asm("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(m));
  return r;
}
__device__ __forceinline__ uint32_t fr_madlo(uint32_t y, uint32_t m, uint32_t t) {
  uint32_t r;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(y), "r"(m), "r"(t));
  return r;
}
template <int N, bool F>
__device__ __forceinline__ uint64_t fr_rotr(uint64_t x) {
  if constexpr (!F) {
    return (x >> N) | (x << (64 - N));
  } else {
    uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32);
    if constexpr (N > 32) { const uint32_t t = lo; lo = hi; hi = t; }
    constexpr int n = N > 32 ? N - 32 : N;
    const uint32_t m = FR_POW2[32 - n];
    return (uint64_t)fr_madlo(lo, m, fr_mulhi(hi, m)) << 32 | fr_madlo(hi, m, fr_mulhi(lo, m));
  }
}
template <int N, bool F>
__device__ __forceinline__ uint64_t fr_shr(uint64_t x) {
  if constexpr (!F) {
    return x >> N;
  } else {
    const uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32), m = FR_POW2[32 - N];
    return (uint64_t)fr_mulhi(hi, m) << 32 | fr_madlo(hi, m, fr_mulhi(lo, m));
  }
}
template <int ROT> __device__ __forceinline__ uint64_t fr_S0(uint64_t a) {
  return fr_rotr<28, (ROT & 1) != 0>(a) ^ fr_rotr<34, (ROT & 1) != 0>(a) ^ fr_rotr<39, (ROT & 1) != 0>(a);
}
template <int ROT> __device__ __forceinline__ uint64_t fr_S1(uint64_t e) {
  return fr_rotr<14, (ROT & 2) != 0>(e) ^ fr_rotr<18, (ROT & 2) != 0>(e) ^ fr_rotr<41, (ROT & 2) != 0>(e);
}
template <int ROT> __device__ __forceinline__ uint64_t fr_s0(uint64_t w) {
  return fr_rotr<1, (ROT & 4) != 0>(w) ^ fr_rotr<8, (ROT & 4) != 0>(w) ^ fr_shr<7, (ROT & 4) != 0>(w);
}
template <int ROT> __device__ __forceinline__ uint64_t fr_s1(uint64_t w) {
  return fr_rotr<19, (ROT & 8) != 0>(w) ^ fr_rotr<61, (ROT & 8) != 0>(w) ^ fr_shr<6, (ROT & 8) != 0>(w);
}
// a 64-bit add NVVM can't reassociate (it otherwise sums h + S1 + Ch + K + W left to right, five IADD.64 deep)
__device__ __forceinline__ uint64_t fr_add(uint64_t x, uint64_t y) {
  uint64_t r;
  asm("add.u64 %0, %1, %2;" : "=l"(r) : "l"(x), "l"(y));
  return r;
}
// one round, its K + W given. As written (ROT & 48 == 0) e's path is SHF, LOP3 and five IADD.64 a round. 16: (h + KW) + Ch
// beside S1, then + S1 and d + T1, four deep in six adds. 32: also d + h + KW ahead, so e' = (d + h + KW + Ch) + S1 is
// three deep, and a' = (T1 + Maj) + S0 three deep from a, in eight adds
template <int ROT>
__device__ __forceinline__ void fr_round(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t& e, uint64_t& f,
                                         uint64_t& g, uint64_t& h, uint64_t kw) {
  if constexpr ((ROT & 32) != 0) {
    const uint64_t hk = fr_add(h, kw), ch = (e & f) ^ (~e & g), s1 = fr_S1<ROT>(e);
    const uint64_t t1 = fr_add(fr_add(hk, ch), s1);
    const uint64_t en = fr_add(fr_add(fr_add(d, hk), ch), s1);
    const uint64_t an = fr_add(fr_add(t1, (a & b) ^ (a & c) ^ (b & c)), fr_S0<ROT>(a));
    h = g; g = f; f = e; e = en; d = c; c = b; b = a; a = an;
  } else if constexpr ((ROT & 16) != 0) {
    const uint64_t t1 = fr_add(fr_add(fr_add(h, kw), (e & f) ^ (~e & g)), fr_S1<ROT>(e));
    const uint64_t t2 = fr_add(fr_S0<ROT>(a), (a & b) ^ (a & c) ^ (b & c));
    h = g; g = f; f = e; e = fr_add(d, t1); d = c; c = b; b = a; a = fr_add(t1, t2);
  } else {
    const uint64_t t1 = h + fr_S1<ROT>(e) + ((e & f) ^ (~e & g)) + kw;
    const uint64_t t2 = fr_S0<ROT>(a) + ((a & b) ^ (a & c) ^ (b & c));
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
}
// s512_compress with 16 rounds unrolled, not 80: the same chain latency (shalat: 3.5-3.6 us a compression at 2.09 GHz, rolled
// or not, bound by one warp's issue rate at ~43 instructions a round) in a fifth of the code
template <int ROT = FR_ROT>
__device__ __forceinline__ void fr_compress(uint64_t st[8], uint64_t w[16]) {
  uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll 1
  for (int r = 0; r < 80; r += 16) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
      if (r) w[j] += fr_s1<ROT>(w[(j + 14) & 15]) + w[(j + 9) & 15] + fr_s0<ROT>(w[(j + 1) & 15]);
      fr_round<ROT>(a, b, c, d, e, f, g, h, SHA512_K[r + j] + w[j]);
    }
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}
__device__ __forceinline__ void fr_pad_block(uint64_t st[8], uint64_t len) {
  uint64_t w[16];
#pragma unroll
  for (int j = 0; j < 16; j++) w[j] = 0;
  w[0] = 0x8000000000000000ull;
  w[15] = len * 8;
  fr_compress(st, w);
}
// sz_segments with fr_compress (the same digests): a lane a segment, a warp's 32 segments' blocks staged through shared
// memory (coalesced), the next block's loads issued before this one's compression. 4 warps a CTA, so a small stage's
// segments take one warp a sub-partition (two issue-bound warps on one each run at half speed)
constexpr int FR_SEG_THREADS = 128;
// the phase probe's clocks (null outside it): fr_tree's thread 0 a CTA stamps FR_TP_SLOTS phases (fr_tp), fr_segments its
// first warp's start and last warp's end (fr_tps[0], [1])
__device__ unsigned long long* fr_tp = nullptr;
__device__ unsigned long long* fr_tps = nullptr;
constexpr int FR_TP_SLOTS = 32;
__device__ __forceinline__ unsigned long long fr_now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// programmatic dependent launch: a segment pass lets fr_tree launch once each of its threads is past its last store (or
// exited); fr_tree reads nothing the pass wrote before fr_pdl_wait (a no-op when launched without the attribute)
__device__ __forceinline__ void fr_pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void fr_pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
extern "C" __global__ void __launch_bounds__(FR_SEG_THREADS) fr_segments(Args a) {
  if (fr_tps && (threadIdx.x & 31) == 0) atomicMin(fr_tps, fr_now());
  extern __shared__ uint32_t fr_sw[];
  const uint8_t* data = (const uint8_t*)a.p[0];
  const uint64_t* mids = (const uint64_t*)a.p[1];
  uint8_t* out = (uint8_t*)a.p[2];
  const long long rows = a.i[0], stride = a.i[1];
  const int s = a.i[2], last = a.i[3];
  const long long total = rows * s;
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t* sh = fr_sw + wid * (32 * 33);
  const long long warp0 = ((long long)blockIdx.x * (blockDim.x >> 5) + wid) * 32;
  if (warp0 >= total) return;
  const long long g = warp0 + lane;
  const int j = (int)(g % s);
  const int bytes = j == s - 1 ? last : 1024;
  // the lane's piece it of each block: 16 bytes at (lane & 7) * 16 of the warp's segment it * 4 + lane / 8
  const uint8_t* src[8];
  int pb[8];
#pragma unroll
  for (int it = 0; it < 8; it++) {
    const long long pg = warp0 + it * 4 + (lane >> 3);
    const int pj = (int)(pg % s);
    pb[it] = pg < total ? (pj == s - 1 ? last : 1024) : 0;
    src[it] = data + (pg / s) * stride + (long long)pj * 1024 + (lane & 7) * 16;
  }
  uint64_t st[8];
  if (g < total)
    for (int q = 0; q < 8; q++) st[q] = mids[8 * j + q];
  uint4 v[8];
#pragma unroll
  for (int it = 0; it < 8; it++) v[it] = pb[it] > 0 ? *(const uint4*)src[it] : make_uint4(0, 0, 0, 0);
#pragma unroll 1
  for (int c = 0; c < 8; c++) {
#pragma unroll
    for (int it = 0; it < 8; it++) {
      uint32_t* d = sh + (it * 4 + (lane >> 3)) * 33 + (lane & 7) * 4;
      d[0] = v[it].x; d[1] = v[it].y; d[2] = v[it].z; d[3] = v[it].w;
    }
    __syncwarp();
    if (c + 1 < 8)
#pragma unroll
      for (int it = 0; it < 8; it++)
        v[it] = (c + 1) * 128 < pb[it] ? *(const uint4*)(src[it] + (c + 1) * 128) : make_uint4(0, 0, 0, 0);
    if (g < total && c * 128 < bytes) {
      uint64_t w[16];
#pragma unroll
      for (int q = 0; q < 16; q++) w[q] = s512_be(sh[lane * 33 + 2 * q], sh[lane * 33 + 2 * q + 1]);
      fr_compress(st, w);
    }
    __syncwarp();
  }
  if (g >= total) return;
  fr_pad_block(st, (uint64_t)(128 + bytes));
  s512_out(st, out + g * 64);
  if (fr_tps) atomicMax(fr_tps + 1, fr_now());
  fr_pdl_trigger();
}
// fr_segments over blocks [c0, c1) of every row (128 bytes each: what a column slab of the producing GEMM wrote), segment-
// major so a warp's 32 rows share a segment and its part of the range. A segment the range starts begins from its prefix
// state; one it ends is padded and its digest written; one it leaves unfinished keeps its state in its digest's slot for
// the next range. p: data, mids, digests   i: rows, stride, s, last, c0, c1, the first segment the launch covers
extern "C" __global__ void __launch_bounds__(FR_SEG_THREADS) fr_segcols(Args a) {
  extern __shared__ uint32_t fr_sw[];
  const uint8_t* data = (const uint8_t*)a.p[0];
  const uint64_t* mids = (const uint64_t*)a.p[1];
  uint8_t* out = (uint8_t*)a.p[2];
  const long long rows = a.i[0], stride = a.i[1];
  const int s = a.i[2], last = a.i[3], c0 = a.i[4], c1 = a.i[5];
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t* sh = fr_sw + wid * (32 * 33);
  const long long wps = (rows + 31) / 32, gw = (long long)blockIdx.x * (blockDim.x >> 5) + wid;
  const int j = a.i[6] + (int)(gw / wps);
  if (j >= s) return;
  const int nbj = (j == s - 1 ? last : 1024) / 128, lo = max(c0 - 8 * j, 0), hi = min(c1 - 8 * j, nbj);
  if (lo >= hi) return;
  const long long r0 = (gw % wps) * 32, r = r0 + lane;
  uint64_t* slot = (uint64_t*)(out + (r * s + j) * 64);
  uint64_t st[8];
  if (r < rows)
    for (int q = 0; q < 8; q++) st[q] = lo == 0 ? mids[8 * j + q] : slot[q];
  // the lane's piece it of each block: 16 bytes at (lane & 7) * 16 of row r0 + it * 4 + lane / 8
  const uint8_t* src[8];
  bool in[8];
#pragma unroll
  for (int it = 0; it < 8; it++) {
    const long long pr = r0 + it * 4 + (lane >> 3);
    in[it] = pr < rows;
    src[it] = data + pr * stride + (long long)j * 1024 + (lane & 7) * 16;
  }
  uint4 v[8];
#pragma unroll
  for (int it = 0; it < 8; it++) v[it] = in[it] ? *(const uint4*)(src[it] + lo * 128) : make_uint4(0, 0, 0, 0);
#pragma unroll 1
  for (int c = lo; c < hi; c++) {
#pragma unroll
    for (int it = 0; it < 8; it++) {
      uint32_t* d = sh + (it * 4 + (lane >> 3)) * 33 + (lane & 7) * 4;
      d[0] = v[it].x; d[1] = v[it].y; d[2] = v[it].z; d[3] = v[it].w;
    }
    __syncwarp();
    if (c + 1 < hi)
#pragma unroll
      for (int it = 0; it < 8; it++) v[it] = in[it] ? *(const uint4*)(src[it] + (c + 1) * 128) : make_uint4(0, 0, 0, 0);
    if (r < rows) {
      uint64_t w[16];
#pragma unroll
      for (int q = 0; q < 16; q++) w[q] = s512_be(sh[lane * 33 + 2 * q], sh[lane * 33 + 2 * q + 1]);
      fr_compress(st, w);
    }
    __syncwarp();
  }
  if (r >= rows) return;
  if (hi == nbj) {
    fr_pad_block(st, (uint64_t)(128 + 128 * nbj));
    s512_out(st, out + (r * s + j) * 64);
  } else {
    for (int q = 0; q < 8; q++) slot[q] = st[q];
  }
  fr_pdl_trigger();
}
// value V (8 big-endian words) into the big-endian message words w = words [base, base + 16) at byte offset o
__device__ __forceinline__ void fr_put(uint64_t w[16], int base, int o, const uint64_t V[8]) {
  const int q = o >> 3, r = (o & 7) * 8;
  uint64_t U[9];
#pragma unroll
  for (int k = 0; k <= 8; k++) U[k] = (k < 8 ? V[k] >> r : 0) | (k > 0 && r ? V[k - 1] << (64 - r) : 0);
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const int d = base + i - q;
    uint64_t u = 0;
#pragma unroll
    for (int k = 0; k <= 8; k++) u = d == k ? U[k] : u;
    w[i] |= u;
  }
}
// fr_put with o and base compile-time constants: the shifts and word selects fold away (the generic one spends ~300
// instructions a value, a twelfth of a compression)
template <int O>
__device__ __forceinline__ void fr_put_c(uint64_t w[16], int base, const uint64_t V[8]) {
  constexpr int q = O >> 3, r = (O & 7) * 8;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const int d = base + i - q;
    if (d >= 0 && d <= 8) {
      uint64_t u = d < 8 ? V[d] >> r : 0;
      if (r && d > 0) u |= V[d - 1] << ((64 - r) & 63);
      w[i] |= u;
    }
  }
}
// a node's children into its block B (B = 1, 2) at the head length o
template <int B>
__device__ __forceinline__ void fr_put_node(uint64_t w[16], int o, const uint64_t L[8], const uint64_t R[8]) {
  if (o == 139) {
    fr_put_c<139>(w, 16 * B, L);
    fr_put_c<211>(w, 16 * B, R);
  } else if (o == 140) {
    fr_put_c<140>(w, 16 * B, L);
    fr_put_c<212>(w, 16 * B, R);
  } else {
    fr_put(w, 16 * B, o, L);
    fr_put(w, 16 * B, o + 72, R);
  }
}
__device__ __forceinline__ void fr_pf(const void* p) { asm volatile("prefetch.global.L1 [%0];" ::"l"(p)); }
// what node idx's hash reads besides its children (the next level's, prefetched while this one hashes)
__device__ __forceinline__ void fr_pf_node(const uint8_t* ntpl, const uint32_t* noff, const uint64_t* nmid, long long idx) {
  fr_pf(ntpl + idx * 384 + 128);
  fr_pf(ntpl + idx * 384 + 256);
  fr_pf(nmid + 8 * idx);
  fr_pf(noff + idx);
}
// fr_hash from the first-block state with a head of O bytes (blocks 1.. unrolled)
template <bool NODE, int O>
__device__ __forceinline__ void fr_hash_mid(const uint8_t* tpl, const uint64_t* mid, const uint64_t L[8], const uint64_t R[8],
                                            uint64_t st[8]) {
  constexpr int nblk = NODE ? 3 : 2;
  if constexpr (NODE) fr_pf(tpl + 256);
#pragma unroll
  for (int k = 0; k < 8; k++) st[k] = __ldg(mid + k);
#pragma unroll
  for (int b = 1; b < nblk; b++) {
    uint64_t w[16];
    const uint4* t = (const uint4*)(tpl + 128 * b);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint4 v = __ldg(t + j);
      w[2 * j] = s512_be(v.x, v.y);
      w[2 * j + 1] = s512_be(v.z, v.w);
    }
    fr_put_c<O>(w, 16 * b, L);
    if constexpr (NODE) fr_put_c<O + 72>(w, 16 * b, R);
    fr_compress(st, w);
  }
}
// SHA-512 of a padded template of nblk blocks, value L at byte o (and R at o + 72, a node's right child), from the IV or (mid
// given) from the state after the first block: a frame leaf's or node's head is longer than a block, so its first block is a
// constant of the tree, its compression done ahead (fr_mid0). The heads' lengths (nodes 139-140 bytes, leaves 169-171) take
// fr_hash_mid
template <bool NODE>
__device__ __forceinline__ void fr_hash(const uint8_t* tpl, const uint64_t* mid, int nblk, int o, const uint64_t L[8],
                                        const uint64_t R[8], uint64_t st[8]) {
  if (mid) {
    if constexpr (NODE) {
      if (o == 139) return fr_hash_mid<true, 139>(tpl, mid, L, R, st);
      if (o == 140) return fr_hash_mid<true, 140>(tpl, mid, L, R, st);
    } else {
      if (o == 169) return fr_hash_mid<false, 169>(tpl, mid, L, R, st);
      if (o == 171) return fr_hash_mid<false, 171>(tpl, mid, L, R, st);
    }
  }
  int b0 = 0;
  if (mid) {
#pragma unroll
    for (int k = 0; k < 8; k++) st[k] = __ldg(mid + k);
    b0 = 1;
  } else {
    s512_iv(st);
  }
#pragma unroll 1
  for (int b = b0; b < nblk; b++) {
    uint64_t w[16];
    const uint4* t = (const uint4*)(tpl + 128 * b);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint4 v = __ldg(t + j);
      w[2 * j] = s512_be(v.x, v.y);
      w[2 * j + 1] = s512_be(v.z, v.w);
    }
    fr_put(w, 16 * b, o, L);
    if (NODE) fr_put(w, 16 * b, o + 72, R);
    fr_compress(st, w);
  }
}
// the state after each of n templates' first block (stride bytes apart) into mid, a thread a template; threads past n
// repeat them (`total` threads: the ahead work of total / n commitments, each tree its own domain)
extern "C" __global__ void fr_mid0(const uint8_t* tpl, int stride, long long n, long long total, uint64_t* mid) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= total) return;
  const long long r = i % n;
  uint64_t st[8], w[16];
  s512_iv(st);
  const uint4* t = (const uint4*)(tpl + r * stride);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint4 v = __ldg(t + j);
    w[2 * j] = s512_be(v.x, v.y);
    w[2 * j + 1] = s512_be(v.z, v.w);
  }
  fr_compress(st, w);
#pragma unroll
  for (int k = 0; k < 8; k++) mid[8 * r + k] = st[k];
}
// p: tables, salts, hm96 mids, my out, cs out, b || c (c written)   i: rows (every layer's slot at once: R x slots), R (row i
// takes salt i % R: the gate's salts stand in for each layer's)
extern "C" __global__ void __launch_bounds__(128) fr_salt(Args a) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= a.i[0]) return;
  const long long ri = i % a.i[1];
  const uint64_t* tables = (const uint64_t*)a.p[0];
  const uint32_t* y = (const uint32_t*)a.p[1] + 48 * ri;
  const uint64_t* mid = (const uint64_t*)a.p[2];
  uint32_t yw[48];
#pragma unroll
  for (int t = 0; t < 48; t++) yw[t] = y[t];
  uint64_t m[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int qq = 0; qq < 48; qq++) {
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const ulonglong2* e = (const ulonglong2*)(tables + ((size_t)(4 * qq + r) * 256 + ((yw[qq] >> (8 * r)) & 0xff)) * 8);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const ulonglong2 v = __ldg(e + k);
        m[2 * k] ^= v.x;
        m[2 * k + 1] ^= v.y;
      }
    }
  }
  uint64_t* my = (uint64_t*)a.p[3] + 8 * i;
#pragma unroll
  for (int k = 0; k < 8; k++) my[k] = m[k];
  uint64_t cs[8], w[16];
#pragma unroll
  for (int t = 0; t < 8; t++) cs[t] = mid[t];
#pragma unroll
  for (int t = 0; t < 16; t++) w[t] = s512_be(yw[2 * t], yw[2 * t + 1]);
  fr_compress(cs, w);
#pragma unroll
  for (int t = 0; t < 8; t++) w[t] = s512_be(yw[32 + 2 * t], yw[32 + 2 * t + 1]);
  w[8] = 0x8000000000000000ull;
#pragma unroll
  for (int t = 9; t < 15; t++) w[t] = 0;
  w[15] = (128 + 192) * 8;
  fr_compress(cs, w);
  uint64_t* co = (uint64_t*)a.p[4] + 8 * i;
#pragma unroll
  for (int k = 0; k < 8; k++) co[k] = cs[k];
  if (i < a.i[1]) s512_out(cs, (uint8_t*)a.p[5] + i * 128 + 64);
}
// one level of a subtree held in sh (big-endian words): threads t < n/2 hash sh[2t], sh[2t+1] into node j0/2 + t of the
// level above, the current level being lw nodes wide at tree offset off with its parents' templates (and first-block states,
// nmid, unless null) from nb
__device__ __noinline__ void fr_level(uint64_t (*sh)[8], int n, long long j0, long long lw, long long off, long long nb,
                                      const uint8_t* ntpl, const uint32_t* noff, const uint64_t* nmid, uint8_t* tree) {
  const int t = threadIdx.x;
  const bool act = t < n / 2;
  uint64_t v[8];
  if (nmid && t < n / 4) fr_pf_node(ntpl, noff, nmid, nb + lw / 2 + j0 / 4 + t);
  if (act) {
    uint64_t L[8], R[8];
#pragma unroll
    for (int k = 0; k < 8; k++) { L[k] = sh[2 * t][k]; R[k] = sh[2 * t + 1][k]; }
    const long long j = j0 / 2 + t, idx = nb + j;
    fr_hash<true>(ntpl + idx * 384, nmid ? nmid + 8 * idx : nullptr, 3, (int)noff[idx], L, R, v);
    s512_out(v, tree + (off + lw + j) * 64);
  }
  __syncthreads();
  if (act)
#pragma unroll
    for (int k = 0; k < 8; k++) sh[t][k] = v[k];
  __syncthreads();
}
// A compression split across a pair of warps, both holding the block w: role 0 the rounds (from st), role 1 the message
// schedule, handing K + W[16..79] over 16 words a lane at a time through the pair's ring of 4 chunk slots; prod and cons
// count the chunks the pair has produced and consumed (shared, zeroed by the caller), c0 the chunks before this compression.
// The warps issue from different sub-partitions, so the rounds' warp issues only the rounds.
__device__ __forceinline__ void fr_compress_pair(uint64_t st[8], const uint64_t w[16], int role, uint64_t (*ring)[16][32],
                                                 volatile int* prod, volatile int* cons, int c0) {
  const int lane = threadIdx.x & 31;
  if (role == 0) {
    uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
    for (int j = 0; j < 16; j++) fr_round<FR_ROT>(a, b, c, d, e, f, g, h, SHA512_K[j] + w[j]);
#pragma unroll 1
    for (int q = 0; q < 4; q++) {
      const int cq = c0 + q;
      while (*prod <= cq) {
      }
      uint64_t kw[16];
#pragma unroll
      for (int j = 0; j < 16; j++) kw[j] = ring[cq & 3][j][lane];
      __threadfence_block();
      __syncwarp();
      if (lane == 0) *cons = cq + 1;
#pragma unroll
      for (int j = 0; j < 16; j++) fr_round<FR_ROT>(a, b, c, d, e, f, g, h, kw[j]);
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
  } else {
    uint64_t x[16];
#pragma unroll
    for (int j = 0; j < 16; j++) x[j] = w[j];
#pragma unroll 1
    for (int q = 0; q < 4; q++) {
      const int cq = c0 + q;
      while (*cons < cq - 3) {
      }
#pragma unroll
      for (int j = 0; j < 16; j++) {
        x[j] += fr_s1<0>(x[(j + 14) & 15]) + x[(j + 9) & 15] + fr_s0<0>(x[(j + 1) & 15]);
        ring[cq & 3][j][lane] = SHA512_K[16 * (q + 1) + j] + x[j];
      }
      __threadfence_block();
      __syncwarp();
      if (lane == 0) *prod = cq + 1;
    }
  }
}
// fr_level with each parent's two remaining blocks hashed by a warp pair (fr_compress_pair): parents 32p + lane by warps 2p
// (rounds) and 2p + 1 (schedule), p < 2; needs the first-block states. ring: 2 pairs' slots (dynamic shared memory), ctr
// their counters (zero on entry, zeroed on exit)
__device__ __noinline__ void fr_level2(uint64_t (*sh)[8], int n, long long j0, long long lw, long long off, long long nb,
                                       const uint8_t* ntpl, const uint32_t* noff, const uint64_t* nmid, uint8_t* tree,
                                       uint64_t (*ring)[4][16][32], int* ctr) {
  const int t = threadIdx.x, p = t >> 6, role = (t >> 5) & 1, lane = t & 31, jl = 32 * p + lane;
  const bool on = 32 * p < n / 2, act = jl < n / 2;
  uint64_t v[8];
  if (role == 0 && jl < n / 4) fr_pf_node(ntpl, noff, nmid, nb + lw / 2 + j0 / 4 + jl);
  if (on) {
    uint64_t L[8], R[8];
    const int js = act ? jl : 0;
#pragma unroll
    for (int k = 0; k < 8; k++) { L[k] = sh[2 * js][k]; R[k] = sh[2 * js + 1][k]; }
    const long long idx = nb + j0 / 2 + js;
    const int o = (int)noff[idx];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = __ldg(nmid + 8 * idx + k);
    fr_pf(ntpl + idx * 384 + 256);
#pragma unroll
    for (int b = 1; b < 3; b++) {
      uint64_t w[16];
      const uint4* tp = (const uint4*)(ntpl + idx * 384 + 128 * b);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const uint4 u = __ldg(tp + j);
        w[2 * j] = s512_be(u.x, u.y);
        w[2 * j + 1] = s512_be(u.z, u.w);
      }
      if (b == 1) fr_put_node<1>(w, o, L, R);
      else fr_put_node<2>(w, o, L, R);
      fr_compress_pair(v, w, role, ring[p], ctr + 2 * p, ctr + 2 * p + 1, 4 * (b - 1));
    }
    if (act && role == 0) s512_out(v, tree + (off + lw + j0 / 2 + jl) * 64);
  }
  __syncthreads();
  if (on && act && role == 0)
#pragma unroll
    for (int k = 0; k < 8; k++) sh[jl][k] = v[k];
  if (t < 4) ctr[t] = 0;
  __syncthreads();
}
// the host's word v at f (mapped host memory), after everything this thread and the CTAs it synchronized with wrote
__device__ __forceinline__ void fr_raise(uint32_t* f, uint32_t v) {
  __threadfence_system();
  asm volatile("st.release.sys.u32 [%0], %1;" ::"l"(f), "r"(v) : "memory");
}
// p: digests, q_mid, my, cs (the layer's slot), hm96 mids, leaf tpl, leaf off, node tpl, node off, hm96 leaves out, b || c
//    (b written), tree, counter, first-block states (R leaves', then width - 1 nodes'), export (mapped host memory), its flag
// i: rows, s, width, first CTA, CTAs in all, export width E, flag value, first-block states on.
//    width / L CTAs of L = min(width, 128) leaves in all, at most 1024, launched whole or in slabs of consecutive CTAs (the
//    rows a slab of the producing GEMM wrote); the last CTA to finish hashes the levels above their roots. With E, the
//    tree stops at the level of E nodes: they go to the export, and the flag is raised for the host to take the levels
//    above.
extern "C" __global__ void __launch_bounds__(128) fr_tree(Args a) {
  __shared__ uint64_t sh[128][8];
  __shared__ int last, ctr[4];
  extern __shared__ uint64_t fr_ring[];  // i[8]: the levels in sh by warp pairs (fr_level2), 32 KB of ring
  const int tid = threadIdx.x, L = blockDim.x, s = a.i[1], total = a.i[4], E = a.i[5];
  const long long cta = blockIdx.x + (long long)a.i[3];
  const long long R = a.i[0], width = a.i[2], i = cta * L + tid;
  const uint8_t* ntpl = (const uint8_t*)a.p[7];
  const uint32_t* noff = (const uint32_t*)a.p[8];
  uint8_t* tree = (uint8_t*)a.p[11];
  const uint64_t* lmid = a.i[7] ? (const uint64_t*)a.p[13] : nullptr;
  const uint64_t* nmid = lmid ? lmid + 8 * R : nullptr;
  uint8_t* ex = (uint8_t*)a.p[14];
  uint32_t* flag = (uint32_t*)a.p[15];
  const bool pair = a.i[8] && nmid && L == 128;
  uint64_t(*ring)[4][16][32] = (uint64_t(*)[4][16][32])fr_ring;
  if (tid < 4) ctr[tid] = 0;
#define FR_TP(k)                                                          \
  do {                                                                    \
    if (fr_tp && tid == 0) fr_tp[cta * FR_TP_SLOTS + (k)] = fr_now();     \
  } while (0)
  FR_TP(0);
  if (i < R) {
    fr_pf((const uint64_t*)a.p[2] + 8 * i);
    fr_pf((const uint64_t*)a.p[3] + 8 * i);
    fr_pf((const uint8_t*)a.p[5] + i * 256 + 128);
    fr_pf((const uint32_t*)a.p[6] + i);
    if (lmid) fr_pf(lmid + 8 * i);
  }
  fr_pdl_wait();
  uint64_t v[8];
  if (i < R) {
    const uint64_t* q = (const uint64_t*)a.p[1];
    uint64_t st[8], w[16];
    const uint4* dg = (const uint4*)((const uint8_t*)a.p[0] + i * (long long)s * 64);
    for (int t = 0; t < s; t += 2) fr_pf(dg + 4 * t);
#pragma unroll
    for (int t = 0; t < 8; t++) st[t] = q[t];
    int j = 0;
#pragma unroll 1
    for (; j + 1 < s; j += 2) {
#pragma unroll
      for (int t = 0; t < 8; t++) {
        const uint4 u = dg[4 * j + t];
        w[2 * t] = s512_be(u.x, u.y);
        w[2 * t + 1] = s512_be(u.z, u.w);
      }
      fr_compress(st, w);
    }
    if (j < s) {
#pragma unroll
      for (int t = 0; t < 4; t++) {
        const uint4 u = dg[4 * j + t];
        w[2 * t] = s512_be(u.x, u.y);
        w[2 * t + 1] = s512_be(u.z, u.w);
      }
      w[8] = 0x8000000000000000ull;
#pragma unroll
      for (int t = 9; t < 15; t++) w[t] = 0;
      w[15] = (uint64_t)(128 + 64 * s) * 8;
      fr_compress(st, w);
    } else {
      fr_pad_block(st, 128 + 64 * (uint64_t)s);
    }
    FR_TP(1);
    const uint64_t* my = (const uint64_t*)a.p[2] + 8 * i;
    const uint64_t* cs = (const uint64_t*)a.p[3] + 8 * i;
    uint64_t* bo = (uint64_t*)((uint8_t*)a.p[10] + i * 128);
    const uint64_t* mid = (const uint64_t*)a.p[4];
    uint64_t lt[8], ym[8];
    // all loads before the stores to bo, which may alias them for the compiler
#pragma unroll
    for (int k = 0; k < 8; k++) {
      ym[k] = __ldg(my + k);
      w[8 + k] = __ldg(cs + k);
      lt[k] = __ldg(mid + 8 + k);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const uint64_t b = s512_bswap(st[k]) ^ ym[k];
      bo[k] = b;
      w[k] = s512_bswap(b);
    }
    fr_compress(lt, w);
    fr_pad_block(lt, 256);
    s512_out(lt, (uint8_t*)a.p[9] + i * 64);
    FR_TP(2);
    fr_hash<false>((const uint8_t*)a.p[5] + i * 256, lmid ? lmid + 8 * i : nullptr, 2, (int)((const uint32_t*)a.p[6])[i], lt,
                   lt, v);
    s512_out(v, tree + i * 64);
    FR_TP(3);
  } else {
    const uint64_t* pd = (const uint64_t*)(tree + i * 64);
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = s512_bswap(pd[k]);
  }
#pragma unroll
  for (int k = 0; k < 8; k++) sh[tid][k] = v[k];
  __syncthreads();
  FR_TP(4);
  long long lw = width, off = 0, nb = 0, j0 = cta * L;
  int n = L, lev = 0;
  for (; n > 1 && !(E && lw <= E); n >>= 1) {
    if (pair) fr_level2(sh, n, j0, lw, off, nb, ntpl, noff, nmid, tree, ring, ctr);
    else fr_level(sh, n, j0, lw, off, nb, ntpl, noff, nmid, tree);
    nb += lw / 2; off += lw; lw /= 2; j0 /= 2;
    FR_TP(5 + lev++);
  }
  const bool here = E && lw <= E;  // this CTA's nodes are the export's
  if (here) {
    if (tid < n) s512_out(sh[tid], ex + (j0 + tid) * 64);
    __threadfence_system();
    __syncthreads();
  }
  if (total == 1) {
    if (here && tid == 0) fr_raise(flag, (uint32_t)a.i[6]);
    FR_TP(27);
    return;
  }
  int* counter = (int*)a.p[12];
  if (tid == 0) {
    __threadfence();
    last = atomicAdd(counter, 1) == total - 1;
  }
  __syncthreads();
  if (!last) return;
  FR_TP(13);
  __threadfence();
  if (here) {
    if (tid == 0) {
      *counter = 0;
      fr_raise(flag, (uint32_t)a.i[6]);
    }
    FR_TP(27);
    return;
  }
  int tl = 0;
  const int n0 = total;  // == lw: the CTAs' subtree roots
  n = n0;
  // levels wider than sh from the tree in memory, the last of them into sh
  for (; n > L; n >>= 1) {
    for (int t = tid; t < n / 2; t += L) {
      uint64_t l[8], r[8], u[8];
      const uint64_t* pd = (const uint64_t*)(tree + (off + 2 * t) * 64);
#pragma unroll
      for (int k = 0; k < 8; k++) { l[k] = s512_bswap(__ldcg(pd + k)); r[k] = s512_bswap(__ldcg(pd + 8 + k)); }
      fr_hash<true>(ntpl + (nb + t) * 384, nmid ? nmid + 8 * (nb + t) : nullptr, 3, (int)noff[nb + t], l, r, u);
      s512_out(u, tree + (off + lw + t) * 64);
      if (n / 2 <= L)
#pragma unroll
        for (int k = 0; k < 8; k++) sh[t][k] = u[k];
    }
    __syncthreads();
    nb += lw / 2; off += lw; lw /= 2;
    FR_TP(15 + tl++);
  }
  if (n == n0 && tid < n0) {
    const uint64_t* pd = (const uint64_t*)(tree + (off + tid) * 64);
#pragma unroll
    for (int k = 0; k < 8; k++) sh[tid][k] = s512_bswap(__ldcg(pd + k));
  }
  __syncthreads();
  FR_TP(14);
  j0 = 0;
  for (; n > 1 && !(E && lw <= E); n >>= 1) {
    if (pair) fr_level2(sh, n, j0, lw, off, nb, ntpl, noff, nmid, tree, ring, ctr);
    else fr_level(sh, n, j0, lw, off, nb, ntpl, noff, nmid, tree);
    nb += lw / 2; off += lw; lw /= 2;
    FR_TP(15 + tl++);
  }
  if (E) {
    if (tid < n) s512_out(sh[tid], ex + tid * 64);
    __threadfence_system();
    __syncthreads();
  }
  if (tid == 0) {
    *counter = 0;
    if (E) fr_raise(flag, (uint32_t)a.i[6]);
  }
  FR_TP(27);
#undef FR_TP
}
// the GEMM's stream waits for the host to write v at f (the draw's coins are in): one thread polls the mapped word; after
// 5 s it gives up and records v in err, so a lost host thread fails the run instead of hanging it
__global__ void host_wait(const uint32_t* f, uint32_t v, uint32_t* err) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  for (;;) {
    uint32_t x;
    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(x) : "l"(f) : "memory");
    if (x == v) return;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t - t0 > 5000000000ull) {
      atomicMax(err, v);
      return;
    }
  }
}
// n round trips through mapped host memory: raise up to i, wait for the host to answer dn = i
__global__ void ping(uint32_t* up, const uint32_t* dn, int n, uint32_t* err) {
  for (int i = 1; i <= n; i++) {
    asm volatile("st.release.sys.u32 [%0], %1;" ::"l"(up), "r"((uint32_t)i) : "memory");
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) {
      uint32_t x;
      asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(x) : "l"(dn) : "memory");
      if (x == (uint32_t)i) break;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
      if (t - t0 > 5000000000ull) {
        *err = i;
        return;
      }
    }
  }
}
static bool g_fast = true;  // HASH=sz: served_zk's hm96_rows.cu as is
static int g_mid = 1;       // the frame leaves' and nodes' first blocks compressed ahead (0: on the critical path, round 1's)
static int g_pair = 0;      // the tree's levels in shared memory hashed by warp pairs (fr_level2)
static int g_pdl = 1;       // fr_tree launched programmatically behind its segment pass (fr_pdl_wait)
// the salt-only parts of `slots` layers' commitments from slot 0, in one launch
static void commit_salt(const Commit& m, int slots, cudaStream_t st) {
  Args a{};
  const unsigned long long ps[6] = {(unsigned long long)g_tables, (unsigned long long)m.salts, (unsigned long long)g_hm_mid,
                                    (unsigned long long)m.my, (unsigned long long)m.cs, (unsigned long long)m.bc};
  for (int j = 0; j < 6; j++) a.p[j] = ps[j];
  const long long rows = (long long)m.R * slots;
  a.i[0] = (int)rows; a.i[1] = m.R;
  fr_salt<<<(unsigned)((rows + 127) / 128), 128, 0, st>>>(a);
  if (g_mid) {
    const long long nl = m.R, nn = m.width - 1;
    fr_mid0<<<(unsigned)((nl * slots + 127) / 128), 128, 0, st>>>(m.ltpl, 256, nl, nl * slots, (uint64_t*)m.fmid);
    if (nn > 0)
      fr_mid0<<<(unsigned)((nn * slots + 127) / 128), 128, 0, st>>>(m.ntpl, 384, nn, nn * slots, (uint64_t*)m.fmid + 8 * nl);
  }
}
// rows [row0, row0 + rows) of the operand (all of it by default)
static void commit_segs(const Commit& m, const uint8_t* data, cudaStream_t st, long long row0 = 0, long long rows = -1) {
  if (rows < 0) rows = m.R;
  Args a{};
  a.p[0] = (unsigned long long)(data + row0 * m.k);
  a.p[1] = (unsigned long long)m.mids;
  a.p[2] = (unsigned long long)(m.digests + row0 * m.s * 64);
  a.i[0] = (int)rows; a.i[1] = m.k; a.i[2] = m.s; a.i[3] = m.last;
  const long long total = rows * m.s;
  fr_segments<<<(unsigned)((total + FR_SEG_THREADS - 1) / FR_SEG_THREADS), FR_SEG_THREADS, FR_SEG_THREADS * 33 * 4, st>>>(a);
}
// a row's 128-byte blocks
static int row_blocks(const Commit& m) { return ((m.s - 1) * 1024 + m.last) / 128; }
// blocks [c0, c1) of every row's row_blocks (fr_segcols): the passes over a row's blocks in order make commit_segs' digests
static void commit_segcols(const Commit& m, const uint8_t* data, cudaStream_t st, int c0, int c1) {
  if (c1 <= c0) return;
  const int j0 = c0 / 8, j1 = std::min(m.s, (c1 + 7) / 8);
  Args a{};
  a.p[0] = (unsigned long long)data;
  a.p[1] = (unsigned long long)m.mids;
  a.p[2] = (unsigned long long)m.digests;
  a.i[0] = m.R; a.i[1] = m.k; a.i[2] = m.s; a.i[3] = m.last; a.i[4] = c0; a.i[5] = c1; a.i[6] = j0;
  const long long threads = (long long)(j1 - j0) * ((m.R + 31) / 32) * 32;
  fr_segcols<<<(unsigned)((threads + FR_SEG_THREADS - 1) / FR_SEG_THREADS), FR_SEG_THREADS, FR_SEG_THREADS * 33 * 4, st>>>(a);
}
// where a tree stops for the host: its level of E nodes into ex, then flag = seq (device views of mapped host memory)
struct Export {
  uint8_t* ex;
  uint32_t* flag;
  int E;
  uint32_t seq;
};
// the level fr_tree exports for an Export's E: the first of at most E nodes, so the leaves themselves when E >= width
static int export_width(long long width, int E) {
  long long lw = width;
  while (lw > E) lw /= 2;
  return (int)lw;
}
// CTAs [cta0, cta0 + n) of the tree's width / 128 (all by default); the last of all to finish takes the top
static void commit_frtree(const Commit& m, int slot, cudaStream_t st, int cta0 = 0, int n = -1, const Export* x = nullptr) {
  const int L = std::min(m.width, 128), total = m.width / L;
  if (n < 0) n = total;
  if (x && export_width(m.width, x->E) > 1024) {
    fprintf(stderr, "an export of %d nodes: the link holds 1024 (link_init)\n", export_width(m.width, x->E));
    exit(2);
  }
  Args f{};
  const unsigned long long ps[16] = {(unsigned long long)m.digests, (unsigned long long)(m.mids + 64 * m.s),
                                     (unsigned long long)(m.my + (size_t)64 * m.R * slot),
                                     (unsigned long long)(m.cs + (size_t)64 * m.R * slot), (unsigned long long)g_hm_mid,
                                     (unsigned long long)m.ltpl, (unsigned long long)m.loff, (unsigned long long)m.ntpl,
                                     (unsigned long long)m.noff, (unsigned long long)m.x, (unsigned long long)m.bc,
                                     (unsigned long long)m.tree, (unsigned long long)m.counter, (unsigned long long)m.fmid,
                                     (unsigned long long)(x ? x->ex : nullptr), (unsigned long long)(x ? x->flag : nullptr)};
  for (int j = 0; j < 16; j++) f.p[j] = ps[j];
  f.i[0] = m.R; f.i[1] = m.s; f.i[2] = m.width; f.i[3] = cta0; f.i[4] = total; f.i[5] = x ? x->E : 0;
  f.i[6] = x ? (int)x->seq : 0; f.i[7] = g_mid; f.i[8] = g_pair && g_mid;
  const unsigned smem = f.i[8] ? 2 * 4 * 16 * 32 * 8 : 0;
  if (!g_pdl) {
    fr_tree<<<(unsigned)n, L, smem, st>>>(f);
    return;
  }
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3((unsigned)n);
  lc.blockDim = dim3((unsigned)L);
  lc.dynamicSmemBytes = smem;
  lc.stream = st;
  lc.attrs = at;
  lc.numAttrs = 1;
  CK(cudaLaunchKernelEx(&lc, fr_tree, f));
}
static void commit_tree(const Commit& m, int slot, const uint8_t* data, cudaStream_t st, const Export* x = nullptr) {
  commit_segs(m, data, st);
  commit_frtree(m, slot, st, 0, -1, x);
}
// slab i of k over m's rows (R / k each, the last taking the rest): its rows [r0, r0 + nr) and the tree's CTAs
// [c0, c0 + nc) whose rows are all in slabs 0..i, since a CTA's leaves read every one of its rows' segment digests (the
// last slab takes every CTA left, the pads' among them)
struct Slab {
  long long r0, nr;
  int c0, nc;
};
static Slab slab(const Commit& m, int k, int i) {
  const int L = std::min(m.width, 128), total = m.width / L;
  const long long rk = m.R / k, r0 = i * rk, r1 = i == k - 1 ? m.R : r0 + rk;
  const int c0 = i == 0 ? 0 : (int)std::min<long long>(total, r0 / L), c1 = i == k - 1 ? total : (int)std::min<long long>(total, r1 / L);
  return {r0, r1 - r0, c0, c1 - c0};
}
static void commit_slab(const Commit& m, int slot, const uint8_t* data, cudaStream_t st, int k, int i, const Export* x) {
  const Slab b = slab(m, k, i);
  if (b.nr > 0) commit_segs(m, data, st, b.r0, b.nr);
  if (b.nc > 0) commit_frtree(m, slot, st, b.c0, b.nc, x);
}
// the next block of a chain from the state alone (so the one-warp and two-warp chains hash the same blocks)
__device__ __forceinline__ void chain_next(const uint64_t st[8], uint64_t w[16]) {
#pragma unroll
  for (int j = 0; j < 16; j++) w[j] = st[j & 7] ^ (0x9E3779B97F4A7C15ull * (j + 1));
}
// n chained compressions a thread, rolled (fr_compress<ROT>) or fully unrolled (s512_compress)
template <int ROT>
__global__ void sha_chain(int n, int unrolled, uint64_t* out) {
  uint64_t st[8], w[16];
  s512_iv(st);
  for (int j = 0; j < 16; j++) w[j] = (uint64_t)(threadIdx.x + 1) * (j + 1);
  for (int i = 0; i < n; i++) {
    if (unrolled) s512_compress(st, w);
    else fr_compress<ROT>(st, w);
    chain_next(st, w);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = st[0];
}
// the same chains, each split across a pair of warps (lane l of warp 0 and of warp 1, on different sub-partitions): warp 0
// the rounds, warp 1 the message schedule, its K + W[16..79] through shared memory 16 words at a time behind named
// barriers 2..5, the block handed over behind barrier 1
template <int ROT>
__global__ void __launch_bounds__(64) sha_chain2(int n, uint64_t* out) {
  __shared__ uint64_t msg[16][32], kw[64][32];
  const int lane = threadIdx.x & 31;
  if (threadIdx.x < 32) {
    uint64_t st[8], w[16];
    s512_iv(st);
    for (int j = 0; j < 16; j++) w[j] = (uint64_t)(lane + 1) * (j + 1);
    for (int i = 0; i < n; i++) {
#pragma unroll
      for (int j = 0; j < 16; j++) msg[j][lane] = w[j];
      __threadfence_block();
      asm volatile("bar.arrive 1, 64;" ::: "memory");
      uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
      for (int j = 0; j < 16; j++) fr_round<ROT>(a, b, c, d, e, f, g, h, SHA512_K[j] + w[j]);
#pragma unroll 1
      for (int q = 1; q < 5; q++) {
        asm volatile("bar.sync %0, 64;" ::"r"(1 + q) : "memory");
#pragma unroll
        for (int j = 0; j < 16; j++) fr_round<ROT>(a, b, c, d, e, f, g, h, kw[16 * (q - 1) + j][lane]);
      }
      st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
      chain_next(st, w);
    }
    out[blockIdx.x * 32 + lane] = st[0];
  } else {
    for (int i = 0; i < n; i++) {
      asm volatile("bar.sync 1, 64;" ::: "memory");
      uint64_t w[16];
#pragma unroll
      for (int j = 0; j < 16; j++) w[j] = msg[j][lane];
#pragma unroll 1
      for (int q = 1; q < 5; q++) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
          w[j] += fr_s1<0>(w[(j + 14) & 15]) + w[(j + 9) & 15] + fr_s0<0>(w[(j + 1) & 15]);
          kw[16 * (q - 1) + j][lane] = SHA512_K[16 * q + j] + w[j];
        }
        __threadfence_block();
        asm volatile("bar.arrive %0, 64;" ::"r"(1 + q) : "memory");
      }
    }
  }
}
// ---------------------------------------------------------------- the host's side: SHA-512, the top of a tree, the draw
static uint64_t g_hk[80];  // SHA512_K, read back from the device
static void (*g_ossl)(void*, const unsigned char*) = nullptr;  // libcrypto's SHA512_Transform: SHA512_CTX's state is its h[8]
static bool g_use_ossl = false;
static inline uint64_t h_rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
static inline uint64_t h_be(const uint8_t* p) {
  uint64_t v = 0;
  for (int k = 0; k < 8; k++) v = v << 8 | p[k];
  return v;
}
static void h_compress_cpp(uint64_t st[8], const uint8_t* blk) {
  uint64_t w[80];
  for (int j = 0; j < 16; j++) w[j] = h_be(blk + 8 * j);
  for (int j = 16; j < 80; j++)
    w[j] = (h_rotr(w[j - 2], 19) ^ h_rotr(w[j - 2], 61) ^ (w[j - 2] >> 6)) + w[j - 7] +
           (h_rotr(w[j - 15], 1) ^ h_rotr(w[j - 15], 8) ^ (w[j - 15] >> 7)) + w[j - 16];
  uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
  for (int j = 0; j < 80; j++) {
    const uint64_t t1 = h + (h_rotr(e, 14) ^ h_rotr(e, 18) ^ h_rotr(e, 41)) + ((e & f) ^ (~e & g)) + g_hk[j] + w[j];
    const uint64_t t2 = (h_rotr(a, 28) ^ h_rotr(a, 34) ^ h_rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}
static void h_compress_ossl(uint64_t st[8], const uint8_t* blk) {
  alignas(64) uint64_t ctx[32] = {0};
  memcpy(ctx, st, 64);
  g_ossl(ctx, blk);
  memcpy(st, ctx, 64);
}
static inline void h_compress(uint64_t st[8], const uint8_t* blk) {
  if (g_use_ossl) h_compress_ossl(st, blk);
  else h_compress_cpp(st, blk);
}
static void h_iv(uint64_t st[8]) {
  const uint64_t iv[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                          0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  memcpy(st, iv, 64);
}
static inline double h_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// a host compression's latency in a chain of n, by the C++ or libcrypto's
static double h_chain_us(bool ossl, int n) {
  g_use_ossl = ossl;
  uint8_t blk[128];
  for (int j = 0; j < 128; j++) blk[j] = (uint8_t)(j * 7 + 1);
  uint64_t st[8];
  h_iv(st);
  std::vector<double> v;
  for (int r = 0; r < 5; r++) {
    const double t0 = h_us();
    for (int i = 0; i < n; i++) {
      h_compress(st, blk);
      blk[i & 127] ^= (uint8_t)st[0];
    }
    v.push_back((h_us() - t0) / n);
  }
  std::sort(v.begin(), v.end());
  return v[2];
}
// SHA512_K from the device; libcrypto's compression if it computes the same block and is faster
static bool g_ossl_equal = false;
static double g_cpp_us = 0, g_ossl_us = 0;
static void h_init() {
  CK(cudaMemcpyFromSymbol(g_hk, SHA512_K, sizeof g_hk));
  for (const char* lib : {"libcrypto.so.3", "libcrypto.so.1.1", "libcrypto.so"})
    if (void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL)) {
      g_ossl = (void (*)(void*, const unsigned char*))dlsym(h, "SHA512_Transform");
      if (g_ossl) break;
    }
  uint8_t blk[128];
  for (int j = 0; j < 128; j++) blk[j] = (uint8_t)(j * 3 + 5);
  uint64_t a[8], b[8];
  h_iv(a);
  h_iv(b);
  h_compress_cpp(a, blk);
  if (g_ossl) {
    h_compress_ossl(b, blk);
    g_ossl_equal = memcmp(a, b, 64) == 0;
  }
  g_cpp_us = h_chain_us(false, 20000);
  g_ossl_us = g_ossl_equal ? h_chain_us(true, 20000) : 0;
  g_use_ossl = g_ossl_equal && g_ossl_us < g_cpp_us;
}
// the nodes' first-block states of a shape, on the host (the same constants fr_mid0 computes)
static std::map<std::pair<int, int>, std::vector<uint64_t>> g_hnmid;
static const std::vector<uint64_t>& h_nmid(const HostConst& c) {
  auto& v = g_hnmid[{c.k, c.R}];
  if (v.empty() && c.width > 1) {
    v.resize((size_t)8 * (c.width - 1));
    for (long long j = 0; j < c.width - 1; j++) {
      h_iv(&v[8 * j]);
      h_compress(&v[8 * j], &c.ntpl[384 * j]);
    }
  }
  return v;
}
// n nodes (64 bytes each, at v) from position j0 of a level lw wide, hashed up to their root. In node-head order the parents
// of a level of lw nodes start at width - lw; a node is its template's last two blocks with L at o and R at o + 72, from its
// first block's state
static void h_node(const HostConst& c, const std::vector<uint64_t>& nmid, long long idx, const uint8_t* L,
                   const uint8_t* R, uint8_t* out) {
  uint8_t m[256];
  uint32_t o;
  memcpy(&o, &c.noff[4 * idx], 4);
  memcpy(m, &c.ntpl[384 * idx + 128], 256);
  memcpy(m + o - 128, L, 64);
  memcpy(m + o + 72 - 128, R, 64);
  uint64_t st[8];
  memcpy(st, &nmid[8 * idx], 64);
  h_compress(st, m);
  h_compress(st, m + 128);
  for (int k = 0; k < 8; k++)
    for (int b = 0; b < 8; b++) out[8 * k + b] = (uint8_t)(st[k] >> (56 - 8 * b));
}
// node idx's template blocks, offset and first-block state into this core's cache
static inline void h_warm_node(const HostConst& c, const std::vector<uint64_t>& nmid, long long idx) {
  const char* t = (const char*)&c.ntpl[384 * idx + 128];
  for (int k = 0; k < 256; k += 64) _mm_prefetch(t + k, _MM_HINT_T0);
  _mm_prefetch(t + 255, _MM_HINT_T0);
  _mm_prefetch((const char*)&c.noff[4 * idx], _MM_HINT_T0);
  _mm_prefetch((const char*)&nmid[8 * idx], _MM_HINT_T0);
}
static void h_sub(const HostConst& c, const std::vector<uint64_t>& nmid, const uint8_t* v, long long n, long long lw,
                  long long j0, uint8_t root[64]) {
  if (n < 1 || j0 < 0 || j0 + n > lw || lw > c.width) {
    fprintf(stderr, "h_sub: nodes [%lld, %lld) of a level of %lld in a tree of %d leaves (an export wider than the tree? "
            "h_top and h_warm take export_width)\n", j0, j0 + n, lw, c.width);
    exit(2);
  }
  static thread_local std::vector<uint8_t> cur, nxt;
  cur.assign(v, v + 64 * n);
  while (n > 1) {
    nxt.resize((size_t)32 * n);
    const long long nb = c.width - lw + j0 / 2;
    for (long long j = 0; j < n / 2; j++) h_node(c, nmid, nb + j, &cur[128 * j], &cur[128 * j + 64], &nxt[64 * j]);
    n /= 2;
    lw /= 2;
    j0 /= 2;
    std::swap(cur, nxt);
  }
  memcpy(root, cur.data(), 64);
}
// P - 1 spinning helpers (P a power of 2). For an export of E >= 2P nodes the caller (w = 0) and each helper w hash a
// subtree of E / P of them to its root, then combine pairwise up the top log2 P levels: at the level of P / h roots, w
// (a multiple of 2h) takes w + h's root once w + h's flag says it is in. Before a step's export arrives, h_warm has each
// thread pull the nodes it will hash into its cache (the six stage shapes take turns, so their templates are cold).
// go = command << 1 | warm, from the caller alone
struct HostPool {
  int P = 1;
  std::vector<std::thread> th;
  alignas(64) std::atomic<uint32_t> go{0};
  uint32_t cmd = 0;
  const HostConst* c = nullptr;
  const std::vector<uint64_t>* nmid = nullptr;
  const uint8_t* ex = nullptr;
  long long E = 0;
  std::atomic<bool> quit{false};
  struct alignas(64) Flag {
    std::atomic<uint32_t> v{0};  // command << 3 | the levels above its subtree w has combined
  };
  Flag ready[64];
  alignas(64) uint8_t roots[64 * 64];
};
static HostPool g_pool;
static void pool_share(int w, bool warm, uint32_t cmd) {
  const HostConst& c = *g_pool.c;
  const std::vector<uint64_t>& nm = *g_pool.nmid;
  const int P = g_pool.P;
  const long long E = g_pool.E, S = E / P;
  if (warm) {
    for (long long n = S, lw = E, j0 = S * w; n > 1; n /= 2, lw /= 2, j0 /= 2)
      for (long long j = 0; j < n / 2; j++) h_warm_node(c, nm, c.width - lw + j0 / 2 + j);
    for (int h = 1; h < P && w % (2 * h) == 0; h *= 2) h_warm_node(c, nm, c.width - P / h + w / (2 * h));
    return;
  }
  uint8_t* mine = g_pool.roots + 64 * w;
  h_sub(c, nm, g_pool.ex + 64 * S * w, S, E, S * w, mine);
  g_pool.ready[w].v.store(cmd << 3, std::memory_order_release);
  for (int h = 1, lv = 1; h < P && w % (2 * h) == 0; h *= 2, lv++) {
    while (g_pool.ready[w + h].v.load(std::memory_order_acquire) < (cmd << 3 | (lv - 1))) _mm_pause();
    uint8_t up[64];
    h_node(c, nm, c.width - P / h + w / (2 * h), mine, g_pool.roots + 64 * (w + h), up);
    memcpy(mine, up, 64);
    g_pool.ready[w].v.store(cmd << 3 | lv, std::memory_order_release);
  }
}
// seen: go when the pool started (read before the helper's thread exists, so it misses no command)
static void pool_worker(int w, uint32_t seen) {
  for (;;) {
    uint32_t g;
    while ((g = g_pool.go.load(std::memory_order_acquire)) == seen) _mm_pause();
    seen = g;
    if (g_pool.quit.load(std::memory_order_acquire)) return;
    pool_share(w, g & 1, g >> 1);
  }
}
static void pool_start(int P) {
  if (P > 64 || (P & (P - 1))) { fprintf(stderr, "a host pool of %d: at most 64, a power of 2\n", P); exit(2); }
  g_pool.P = P;
  g_pool.quit = false;
  const uint32_t g0 = g_pool.go.load(std::memory_order_acquire);
  for (int w = 1; w < P; w++) g_pool.th.emplace_back(pool_worker, w, g0);
}
static void pool_stop() {
  g_pool.quit = true;
  g_pool.go.store(++g_pool.cmd << 1, std::memory_order_release);
  for (auto& t : g_pool.th) t.join();
  g_pool.th.clear();
  g_pool.P = 1;
}
// warm the pool's caches for an Export's E on shape c (h_top's next call, same c and E)
static void h_warm(const HostConst& c, const std::vector<uint64_t>& nmid, int E) {
  E = export_width(c.width, E);
  if (g_pool.P == 1 || E < 2 * g_pool.P) {
    for (long long n = E, lw = E; n > 1; n /= 2, lw /= 2)
      for (long long j = 0; j < n / 2; j++) h_warm_node(c, nmid, c.width - lw + j);
    return;
  }
  g_pool.c = &c;
  g_pool.nmid = &nmid;
  g_pool.E = E;
  const uint32_t cmd = ++g_pool.cmd;
  g_pool.go.store(cmd << 1 | 1, std::memory_order_release);
  pool_share(0, true, cmd);
}
// the levels above the level fr_tree exported (ex) for an Export's E to the root, by the pool when it has helpers and that
// level has at least 2P nodes (h_nmid(c) computed before: the helpers read it)
static void h_top(const HostConst& c, const std::vector<uint64_t>& nmid, const uint8_t* ex, int E, uint8_t root[64]) {
  E = export_width(c.width, E);
  const int P = g_pool.P;
  if (P == 1 || E < 2 * P) return h_sub(c, nmid, ex, E, E, 0, root);
  // a helper still warming reads c, nmid and E: rewritten only when they change (never right after h_warm)
  if (g_pool.c != &c) g_pool.c = &c;
  if (g_pool.nmid != &nmid) g_pool.nmid = &nmid;
  if (g_pool.E != E) g_pool.E = E;
  g_pool.ex = ex;
  const uint32_t cmd = ++g_pool.cmd;
  g_pool.go.store(cmd << 1, std::memory_order_release);
  pool_share(0, false, cmd);
  memcpy(root, g_pool.roots, 64);
}
// mapped host memory the GPU and the host's thread meet in: the exported level, the GPU's flag (up) and the host's (dn)
struct Link {
  uint8_t *ex_h = nullptr, *ex_d = nullptr;
  uint32_t *up_h = nullptr, *up_d = nullptr, *dn_h = nullptr, *dn_d = nullptr, *err_d = nullptr;
};
static Link g_link;
static void link_init() {
  if (g_link.ex_h) return;
  void* p;
  CK(cudaHostAlloc(&p, 64 * 1024 + 256, cudaHostAllocMapped));
  memset(p, 0, 64 * 1024 + 256);
  g_link.ex_h = (uint8_t*)p;
  g_link.up_h = (uint32_t*)(g_link.ex_h + 64 * 1024);
  g_link.dn_h = g_link.up_h + 32;
  void* q;
  CK(cudaHostGetDevicePointer(&q, p, 0));
  g_link.ex_d = (uint8_t*)q;
  g_link.up_d = (uint32_t*)(g_link.ex_d + 64 * 1024);
  g_link.dn_d = g_link.up_d + 32;
  CK(cudaMalloc(&g_link.err_d, sizeof(uint32_t)));
  CK(cudaMemset(g_link.err_d, 0, sizeof(uint32_t)));
}
static inline uint32_t h_load(const uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void h_store(uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
// one stage's commitment as the host sees it: the shape, and the export's width (1: the GPU's root, nothing to hash)
struct HostStep {
  const HostConst* c;
  int E;
};
struct HostStats {
  long steps = 0, timeouts = 0;
  double top_us = 0, top_max_us = 0;
};
// the host's side of `iters` passes over a step list: warm the caches for step q's top, wait for its flag (q + 1), hash
// the levels above its export, wait draw_us (the verifier's round trip for the coins), then raise the GEMM's flag; stops
// after 5 s without a flag
static void host_loop(std::vector<HostStep> steps, int iters, int draw_us, HostStats* out) {
  uint8_t root[64];
  for (int it = 0; it < iters; it++)
    for (size_t q = 0; q < steps.size(); q++) {
      const uint32_t v = (uint32_t)q + 1;
      if (steps[q].E > 1) h_warm(*steps[q].c, h_nmid(*steps[q].c), steps[q].E);
      const double t0 = h_us();
      while (h_load(g_link.up_h) != v)
        if (h_us() - t0 > 5e6) {
          out->timeouts++;
          return;
        }
      const double t1 = h_us();
      if (steps[q].E > 1) h_top(*steps[q].c, h_nmid(*steps[q].c), g_link.ex_h, steps[q].E, root);
      const double t2 = h_us();
      out->top_us += t2 - t1;
      out->top_max_us = std::max(out->top_max_us, t2 - t1);
      out->steps++;
      while (h_us() - t2 < draw_us) {
      }
      h_store(g_link.dn_h, v);
    }
}
// the latency of one compression in a chain: (n = 257) - (n = 1) over 256, by threads a CTA (1: one thread on the GPU), for
// each rotation choice and the two-warp split; the host's compression and the GPU-host round trip through mapped memory
static void sha_latency(Dev& d) {
  uint64_t *out, *out2;
  CK(cudaMalloc(&out, 128 * 188 * sizeof(uint64_t)));
  CK(cudaMalloc(&out2, 128 * 188 * sizeof(uint64_t)));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto per = [&](const std::function<void(int)>& launch) {
    double ms[2];
    for (int k = 0; k < 2; k++) {
      const int n = k ? 257 : 1;
      std::vector<double> v;
      for (int r = 0; r < 5; r++) {
        CK(cudaEventRecord(e0, d.st));
        launch(n);
        CK(cudaEventRecord(e1, d.st));
        CK(cudaEventSynchronize(e1));
        float x;
        CK(cudaEventElapsedTime(&x, e0, e1));
        v.push_back(x);
      }
      std::sort(v.begin(), v.end());
      ms[k] = v[2];
    }
    return (ms[1] - ms[0]) * 1e3 / 256;
  };
  printf("{\"mode\": \"sha_latency\", \"mhz\": %.0f, \"us_a_compression\": {", nvml_mhz(d));
  bool first = true;
  auto put = [&](const std::string& nm, double us) {
    printf("%s\"%s\": %.3f", first ? "" : ", ", nm.c_str(), us);
    first = false;
  };
  using Chain = void (*)(int, int, uint64_t*);
  const std::pair<int, Chain> rolled[] = {{0, sha_chain<0>}, {1, sha_chain<1>}, {2, sha_chain<2>}, {3, sha_chain<3>},
                                          {4, sha_chain<4>}, {8, sha_chain<8>}, {12, sha_chain<12>},
                                          {16, sha_chain<16>}, {32, sha_chain<32>}};
  for (const auto& rc : rolled)
    for (int thr : {1, 32}) put("rolled_r" + std::to_string(rc.first) + "_" + std::to_string(thr), per([&](int n) {
      rc.second<<<1, thr, 0, d.st>>>(n, 0, out);
    }));
  put("unrolled_32", per([&](int n) { sha_chain<0><<<1, 32, 0, d.st>>>(n, 1, out); }));
  using Chain2 = void (*)(int, uint64_t*);
  const std::pair<int, Chain2> split[] = {{0, sha_chain2<0>}, {1, sha_chain2<1>}, {2, sha_chain2<2>}, {3, sha_chain2<3>},
                                          {16, sha_chain2<16>}, {32, sha_chain2<32>}};
  for (const auto& sc : split) put("split2_r" + std::to_string(sc.first) + "_32", per([&](int n) {
    sc.second<<<1, 64, 0, d.st>>>(n, out2);
  }));
  // the split's chains, and the reassociated rounds', equal the one-warp chains'
  sha_chain<0><<<1, 32, 0, d.st>>>(257, 0, out);
  std::vector<uint64_t> h1(32), h2(32);
  CK(cudaMemcpy(h1.data(), out, 256, cudaMemcpyDeviceToHost));
  bool same = true;
  for (const auto& sc : split) {
    sc.second<<<1, 64, 0, d.st>>>(257, out2);
    CK(cudaMemcpy(h2.data(), out2, 256, cudaMemcpyDeviceToHost));
    same &= h1 == h2;
  }
  for (const auto& rc : rolled) {
    rc.second<<<1, 32, 0, d.st>>>(257, 0, out2);
    CK(cudaMemcpy(h2.data(), out2, 256, cudaMemcpyDeviceToHost));
    same &= h1 == h2;
  }
  printf("}, \"split_equal\": %s}\n", same ? "true" : "false");
  fflush(stdout);
  // the host: a compression's latency in a chain (C++ and libcrypto's, h_init), the round trip through mapped memory
  link_init();
  const int n = 2000;
  h_store(g_link.up_h, 0);
  h_store(g_link.dn_h, 0);
  std::thread echo([&] {
    for (int i = 1; i <= n; i++) {
      const double t0 = h_us();
      while (h_load(g_link.up_h) != (uint32_t)i)
        if (h_us() - t0 > 5e6) return;
      h_store(g_link.dn_h, (uint32_t)i);
    }
  });
  const double t0 = h_us();
  ping<<<1, 1, 0, d.st>>>(g_link.up_d, g_link.dn_d, n, g_link.err_d);
  CK(cudaStreamSynchronize(d.st));
  const double rt = (h_us() - t0) / n;
  echo.join();
  uint32_t err = 0;
  CK(cudaMemcpy(&err, g_link.err_d, 4, cudaMemcpyDeviceToHost));
  printf("{\"mode\": \"cpu_latency\", \"cpp_us_a_compression\": %.4f, \"ossl_us_a_compression\": %.4f, \"ossl_equal\": %s, "
         "\"host_uses\": \"%s\", \"mapped_round_trip_us\": %.3f, \"round_trip_errors\": %u}\n",
         g_cpp_us, g_ossl_us, g_ossl_equal ? "true" : "false", g_use_ossl ? "ossl" : "cpp", rt, err);
  fflush(stdout);
  CK(cudaFree(out));
  CK(cudaFree(out2));
}
// the commitment of the gate's rows into poisoned buffers by path (1 hm96_rows.cu, 2 fast; 0 the no-write control) equals
// Python's leaves, b || c and root; the whole tree is copied out for the paths to be compared node by node. The fast path
// in k slabs of rows (a launch pair a slab, as behind a slabbed GEMM), and with E stopped at the level of E nodes for the
// host to hash the top (the root then the host's, the flag checked)
static bool commit_gate(const Commit& m, int path, std::vector<uint8_t>* tree_out = nullptr, int k = 1, int E = 0,
                        int cols = 0) {
  const HostConst& c = *m.hc;
  uint8_t* rows = dput(c.rows.data(), c.rows.size());
  CK(cudaMemset(m.x, 0xA5, (size_t)64 * m.R));
  CK(cudaMemset(m.bc, 0xA5, (size_t)128 * m.R));
  CK(cudaMemset(m.tree, 0xA5, m.tree_bytes));
  CK(cudaMemset(m.my, 0xA5, (size_t)64 * m.R * LAYERS));
  CK(cudaMemset(m.cs, 0xA5, (size_t)64 * m.R * LAYERS));
  CK(cudaMemset(m.fmid, 0xA5, (size_t)64 * (m.R + m.width - 1)));
  if (!c.pads.empty()) CK(cudaMemcpy(m.tree + (size_t)64 * c.R, c.pads.data(), c.pads.size(), cudaMemcpyHostToDevice));
  Export x{g_link.ex_d, g_link.up_d, E, 0x5A5A0000u + (uint32_t)E};
  if (E) {
    memset(g_link.ex_h, 0xA5, 64 * 1024);
    h_store(g_link.up_h, 0);
  }
  if (path == 1) commit_run(m, rows, 0);
  if (path == 2 && cols) {
    commit_salt(m, LAYERS, 0);
    CK(cudaMemset(m.digests, 0xA5, (size_t)64 * m.R * m.s));
    const int B = row_blocks(m);
    for (int i = 0; i < cols; i++) commit_segcols(m, rows, 0, i * B / cols, (i + 1) * B / cols);
    commit_frtree(m, LAYERS - 1, 0, 0, -1, E ? &x : nullptr);
  } else if (path == 2) {
    commit_salt(m, LAYERS, 0);
    for (int i = 0; i < k; i++) commit_slab(m, LAYERS - 1, rows, 0, k, i, E ? &x : nullptr);
  }
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint8_t> hl((size_t)64 * m.R), bc((size_t)128 * m.R), root(64);
  CK(cudaMemcpy(hl.data(), m.x, hl.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(bc.data(), m.bc, bc.size(), cudaMemcpyDeviceToHost));
  if (E) {
    if (h_load(g_link.up_h) != x.seq) return false;
    h_top(c, h_nmid(c), g_link.ex_h, E, root.data());
  } else {
    CK(cudaMemcpy(root.data(), m.tree + m.tree_bytes - 64, 64, cudaMemcpyDeviceToHost));
  }
  if (tree_out) {
    tree_out->resize(m.tree_bytes);
    CK(cudaMemcpy(tree_out->data(), m.tree, m.tree_bytes, cudaMemcpyDeviceToHost));
  }
  CK(cudaFree(rows));
  return hl == c.hl && bc == c.bc && root == c.root;
}

// ---------------------------------------------------------------- the draw: one thread waits on the global timer
__global__ void draw_wait(unsigned long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - t0 < ns);
}

// ---------------------------------------------------------------- forming: dst = src + noise, a pass over the operand
__device__ __forceinline__ uint32_t mix32(uint64_t x) {
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29; x *= 0x94D049BB133111EBull; x ^= x >> 32;
  return (uint32_t)x;
}
__global__ void form_add(const int4* __restrict__ src, int4* __restrict__ dst, size_t n16, uint64_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = __ldcs(src + i);
    const uint64_t h = (i + 1) * 0x9E3779B97F4A7C15ull ^ seed;
    v.x = (int)__vadd4((unsigned)v.x, mix32(h) & 0x01010101u);
    v.y = (int)__vadd4((unsigned)v.y, mix32(h + 1) & 0x01010101u);
    v.z = (int)__vadd4((unsigned)v.z, mix32(h + 2) & 0x01010101u);
    v.w = (int)__vadd4((unsigned)v.w, mix32(h + 3) & 0x01010101u);
    __stcs(dst + i, v);
  }
}
// P = the s8 quantization of S (a stand-in for exp-quantize's traffic: 4 bytes read, 1 written an entry)
__global__ void soft_q(const int4* __restrict__ S, int* __restrict__ P, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    const int4 s = __ldcs(S + i);
    const unsigned b = ((unsigned)(s.x >> 12) & 0xff) | (((unsigned)(s.y >> 12) & 0xff) << 8) |
                       (((unsigned)(s.z >> 12) & 0xff) << 16) | (((unsigned)(s.w >> 12) & 0xff) << 24);
    P[i] = (int)b;
  }
}
static void form(const void* src, void* dst, size_t bytes, uint64_t seed, cudaStream_t st, int sms) {
  form_add<<<sms * 8, 256, 0, st>>>((const int4*)src, (int4*)dst, bytes / 16, seed);
}

// ---------------------------------------------------------------- the deferred pass's attention, fused (ATT=fused)
// One sequence's three attention products against its cache, nothing S-sized in memory: a CTA is up to 128 stacked query
// rows (4 query heads x tokens) of one KV head, a warp 16 of them, over 32-key tiles of K and V (cp.async, double-buffered):
//   S = Q K^T (clean, to rebuild P), P = att_q(S) in registers, P' = P + noise (bytewise), O = P' V, and Sn = Qn K^T (Qn the
//   noised Q, formed beforehand) reduced to a checksum a row (a stand-in for what consumes it).
// mma.sync m16n8k32 s8. QK^T's n-tile a column g is key 16(a>>1) + 4(g>>1) + 2(a&1) + (g&1) of the tile, so that S's
// accumulators are P's A fragments for P.V in the keys' natural order (no shuffles). K's 16-byte chunks are swizzled by
// c ^ ((key & 1) | ((key >> 1) & 6)), V's rows padded to 48 bytes: conflict-free fragment loads.
// Q, Qn [HKV][rows][HD], K [HKV][lk][HD], V [HKV][HD][lk], O [HKV][rows][HD] s32, csum [HKV][rows]; lk % 32 == 0.
__host__ __device__ __forceinline__ int att_q(int s) {
  const int x = s >> 10;
  return x < 0 ? 0 : x > 127 ? 127 : x;
}
__host__ __device__ __forceinline__ uint32_t att_noise(uint32_t seed, uint32_t r, uint32_t k0) {
  uint32_t x = seed ^ (r * 0x9E3779B1u) ^ (k0 * 0x85EBCA77u);
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15;
  return x & 0x3F3F3F3Fu;
}
__device__ __forceinline__ void cp_async16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void mma_s8(int c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack4(int a, int b, int c, int d) {
  return (uint32_t)att_q(a) | ((uint32_t)att_q(b) << 8) | ((uint32_t)att_q(c) << 16) | ((uint32_t)att_q(d) << 24);
}
constexpr int ATT_KT = 32, ATT_VS = 48;
__global__ void __launch_bounds__(256) att_fused(const int8_t* __restrict__ Q, const int8_t* __restrict__ Qn,
                                                 const int8_t* __restrict__ K, const int8_t* __restrict__ V, int rows, int lk,
                                                 int* __restrict__ O, int* __restrict__ csum, uint32_t seed, int ns) {
  __shared__ __align__(16) uint8_t ks[2][ATT_KT * HD];
  __shared__ __align__(16) uint8_t vs[2][HD * ATT_VS];
  const int head = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int nthr = blockDim.x;
  const int r0 = blockIdx.x * 128 + warp * 16;
  const int8_t* Kh = K + (size_t)head * lk * HD;
  const int8_t* Vh = V + (size_t)head * HD * lk;
  auto load = [&](int kt, int buf) {
    const int key0 = kt * ATT_KT;
    for (int c = threadIdx.x; c < ATT_KT * 8; c += nthr) {
      const int key = c >> 3, ch = c & 7, sw = (key & 1) | ((key >> 1) & 6);
      cp_async16((uint32_t)__cvta_generic_to_shared(&ks[buf][key * HD + ((ch ^ sw) << 4)]), Kh + (size_t)(key0 + key) * HD + ch * 16);
    }
    for (int c = threadIdx.x; c < HD * 2; c += nthr) {
      const int dim = c >> 1, ch = c & 1;
      cp_async16((uint32_t)__cvta_generic_to_shared(&vs[buf][dim * ATT_VS + ch * 16]), Vh + (size_t)dim * lk + key0 + ch * 16);
    }
    cp_commit();
  };
  uint32_t qa[4][4], qn[4][4];
  const int8_t* Qh = Q + (size_t)head * rows * HD;
  const int8_t* Qnh = Qn + (size_t)head * rows * HD;
#pragma unroll
  for (int kk = 0; kk < 4; kk++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = r0 + g + 8 * (j & 1), col = 32 * kk + 16 * (j >> 1) + 4 * t;
      qa[kk][j] = r < rows ? *(const uint32_t*)(Qh + (size_t)r * HD + col) : 0u;
      qn[kk][j] = r < rows ? *(const uint32_t*)(Qnh + (size_t)r * HD + col) : 0u;
    }
  int o[16][4];
#pragma unroll
  for (int nd = 0; nd < 16; nd++) o[nd][0] = o[nd][1] = o[nd][2] = o[nd][3] = 0;
  int cs0 = 0, cs1 = 0;
  const int nt = lk / ATT_KT, kb = (int)((long long)nt * blockIdx.z / ns), ke = (int)((long long)nt * (blockIdx.z + 1) / ns);
  if (kb < ke) load(kb, 0);
#pragma unroll 1
  for (int kt = kb; kt < ke; kt++) {
    const int buf = (kt - kb) & 1;
    if (kt + 1 < ke) load(kt + 1, buf ^ 1);
    else cp_commit();
    cp_wait<1>();
    __syncthreads();
    int s[4][4], sn[4][4];
#pragma unroll
    for (int a = 0; a < 4; a++) s[a][0] = s[a][1] = s[a][2] = s[a][3] = sn[a][0] = sn[a][1] = sn[a][2] = sn[a][3] = 0;
#pragma unroll
    for (int a = 0; a < 4; a++) {
      const int key = 16 * (a >> 1) + 4 * (g >> 1) + 2 * (a & 1) + (g & 1), sw = (key & 1) | ((key >> 1) & 6);
      const uint8_t* kr = &ks[buf][key * HD + 4 * t];
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        const uint32_t b0 = *(const uint32_t*)(kr + (((2 * kk) ^ sw) << 4)), b1 = *(const uint32_t*)(kr + (((2 * kk + 1) ^ sw) << 4));
        mma_s8(s[a], qa[kk], b0, b1);
        mma_s8(sn[a], qn[kk], b0, b1);
      }
    }
#pragma unroll
    for (int a = 0; a < 4; a++) { cs0 += sn[a][0] + sn[a][1]; cs1 += sn[a][2] + sn[a][3]; }
    const uint32_t k0 = (uint32_t)(kt * ATT_KT);
    uint32_t p[4];
    p[0] = __vadd4(pack4(s[0][0], s[0][1], s[1][0], s[1][1]), att_noise(seed, r0 + g, k0 + 4 * t));
    p[1] = __vadd4(pack4(s[0][2], s[0][3], s[1][2], s[1][3]), att_noise(seed, r0 + g + 8, k0 + 4 * t));
    p[2] = __vadd4(pack4(s[2][0], s[2][1], s[3][0], s[3][1]), att_noise(seed, r0 + g, k0 + 16 + 4 * t));
    p[3] = __vadd4(pack4(s[2][2], s[2][3], s[3][2], s[3][3]), att_noise(seed, r0 + g + 8, k0 + 16 + 4 * t));
#pragma unroll
    for (int nd = 0; nd < 16; nd++) {
      const uint8_t* vr = &vs[buf][(8 * nd + g) * ATT_VS + 4 * t];
      mma_s8(o[nd], p, *(const uint32_t*)vr, *(const uint32_t*)(vr + 16));
    }
    __syncthreads();
  }
  cp_wait<0>();
  int* Oh = O + (size_t)head * rows * HD;
#pragma unroll
  for (int nd = 0; nd < 16; nd++) {
    const int col = 8 * nd + 2 * t;
    int* o0 = Oh + (size_t)(r0 + g) * HD + col;
    int* o1 = Oh + (size_t)(r0 + g + 8) * HD + col;
    if (ns == 1) {
      if (r0 + g < rows) *(int2*)o0 = make_int2(o[nd][0], o[nd][1]);
      if (r0 + g + 8 < rows) *(int2*)o1 = make_int2(o[nd][2], o[nd][3]);
    } else {
      if (r0 + g < rows) { atomicAdd(o0, o[nd][0]); atomicAdd(o0 + 1, o[nd][1]); }
      if (r0 + g + 8 < rows) { atomicAdd(o1, o[nd][2]); atomicAdd(o1 + 1, o[nd][3]); }
    }
  }
  cs0 += __shfl_xor_sync(0xffffffffu, cs0, 1); cs0 += __shfl_xor_sync(0xffffffffu, cs0, 2);
  cs1 += __shfl_xor_sync(0xffffffffu, cs1, 1); cs1 += __shfl_xor_sync(0xffffffffu, cs1, 2);
  if (t == 0) {
    int* c = csum + (size_t)head * rows + r0 + g;
    if (r0 + g < rows) { if (ns == 1) c[0] = cs0; else atomicAdd(c, cs0); }
    if (r0 + g + 8 < rows) { if (ns == 1) c[8] = cs1; else atomicAdd(c + 8, cs1); }
  }
}
// ns key splits a launch (0: the count, of at least 8 key tiles a split, whose CTAs fill their last wave of resident CTAs
// best, fewer splits unless 2% better); split, a CTA adds its keys' share of O and the checksums into zeroed outputs
// (integer sums: the same bits whatever the split)
static void att_run(const int8_t* Q, const int8_t* Qn, const int8_t* K, const int8_t* V, int rows, int lk, int* O, int* csum,
                    uint32_t seed, cudaStream_t st, int ns = 0) {
  static int sms = 0, occ[9] = {0};
  if (!sms) CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const int warps = std::min(8, (rows + 15) / 16), rb = (rows + 127) / 128, nt = lk / ATT_KT;
  if (!occ[warps]) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[warps], att_fused, 32 * warps, 0));
  if (ns <= 0) {
    const long long slots = (long long)sms * std::max(1, occ[warps]), base = (long long)rb * HKV;
    double best = -1;
    for (int s = 1; s <= std::max(1, nt / 8); s++) {
      const long long n = base * s;
      const double eff = (double)n / ((n + slots - 1) / slots * slots);
      if (eff > best + 0.02) { best = eff; ns = s; }
    }
  }
  if (ns > 1) {
    CK(cudaMemsetAsync(O, 0, (size_t)HKV * rows * HD * sizeof(int), st));
    CK(cudaMemsetAsync(csum, 0, (size_t)HKV * rows * sizeof(int), st));
  }
  att_fused<<<dim3((unsigned)rb, HKV, (unsigned)ns), 32 * warps, 0, st>>>(Q, Qn, K, V, rows, lk, O, csum, seed, ns);
}

// ---------------------------------------------------------------- cuBLASLt int8, strided batched, verified on batch 0
struct LtB {
  Lt L;
  int m, n, k, batch;
};
static LtB lt_make_b(int m, int n, int k, int batch, long long sa, long long sw, long long sd, const void* A, const void* W,
                     void* D, cudaStream_t st) {
  LtB B;
  B.m = m; B.n = n; B.k = k; B.batch = batch;
  Lt& L = B.L;
  CL(cublasLtMatmulDescCreate(&L.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CL(cublasLtMatrixLayoutCreate(&L.la, CUDA_R_8I, k, n, k));
  CL(cublasLtMatrixLayoutCreate(&L.lb, CUDA_R_8I, k, m, k));
  CL(cublasLtMatrixLayoutCreate(&L.ld, CUDA_R_32I, n, m, n));
  if (batch > 1) {
    const int32_t bc = batch;
    CL(cublasLtMatrixLayoutSetAttribute(L.la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof bc));
    CL(cublasLtMatrixLayoutSetAttribute(L.lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof bc));
    CL(cublasLtMatrixLayoutSetAttribute(L.ld, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof bc));
    CL(cublasLtMatrixLayoutSetAttribute(L.la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sw, sizeof sw));
    CL(cublasLtMatrixLayoutSetAttribute(L.lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof sa));
    CL(cublasLtMatrixLayoutSetAttribute(L.ld, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sd, sizeof sd));
  }
  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_wsz, sizeof g_wsz));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  if (cublasLtMatmulAlgoGetHeuristic(g_lt, L.op, L.la, L.lb, L.ld, L.ld, pref, 16, res, &got) != CUBLAS_STATUS_SUCCESS) got = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float best = 1e30f;
  int bi = -1;
  const int32_t one = 1, zero = 0;
  const int mr = std::min(m, 256), rs = m / mr;
  const size_t nchk = (size_t)mr * n;
  ref_rows<<<(unsigned)((nchk + 255) / 256), 256, 0, st>>>((const int8_t*)A, (const int8_t*)W, g_ref, mr, rs, n, k);
  for (int i = 0; i < got; i++) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    CK(cudaMemsetAsync(D, 0xA5, (size_t)m * n * sizeof(int), st));
    if (cublasLtMatmul(g_lt, L.op, &one, W, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, g_ws, g_wsz, st) !=
        CUBLAS_STATUS_SUCCESS)
      continue;
    CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, st));
    count_neq_rows<<<(unsigned)((nchk + 255) / 256), 256, 0, st>>>((const int*)D, g_ref, mr, rs, n, g_cnt);
    unsigned long long bad = 0;
    CK(cudaMemcpyAsync(&bad, g_cnt, sizeof bad, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    if (bad) { L.rejected++; continue; }
    CK(cudaEventRecord(e0, st));
    for (int r = 0; r < 3; r++)
      cublasLtMatmul(g_lt, L.op, &one, W, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, g_ws, g_wsz, st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) { best = ms; bi = i; }
  }
  CK(cudaGetLastError());
  if (bi >= 0) {
    L.algo = res[bi].algo;
    L.ms = best / 3;
    size_t wr;
    cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_ID, &L.algo_id, sizeof L.algo_id, &wr);
  } else {
    fprintf(stderr, "cuBLASLt int8: no algorithm ran for m %d n %d k %d batch %d\n", m, n, k, batch);
  }
  CL(cublasLtMatmulPreferenceDestroy(pref));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return B;
}
// a handle and workspace per stream that may run a GEMM at the same time as another
struct LtCtx {
  cublasLtHandle_t h;
  void* ws;
};
static LtCtx lt_ctx() {
  LtCtx c;
  CL(cublasLtCreate(&c.h));
  CK(cudaMalloc(&c.ws, g_wsz));
  return c;
}
static void ltb_run(const LtB& B, const LtCtx& c, const void* A, const void* W, void* D, cudaStream_t st) {
  const int32_t one = 1, zero = 0;
  if (B.L.algo_id == -2) return;
  CL(cublasLtMatmul(c.h, B.L.op, &one, W, B.L.la, A, B.L.lb, &zero, D, B.L.ld, D, B.L.ld, &B.L.algo, c.ws, g_wsz, st));
}
static std::map<std::tuple<int, int, int, int>, LtB> g_plans;
static int g_rejected_total = 0, g_missing = 0;
static const LtB& plan(int m, int n, int k, int batch, long long sa, long long sw, long long sd, const void* A, const void* W,
                       void* D, cudaStream_t st) {
  const auto key = std::make_tuple(m, n, k, batch);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  LtB b = lt_make_b(m, n, k, batch, sa, sw, sd, A, W, D, st);
  g_rejected_total += b.L.rejected;
  g_missing += b.L.algo_id == -2;
  return g_plans.emplace(key, b).first->second;
}

// ---------------------------------------------------------------- timing helpers
struct Graph {
  cudaGraphExec_t ge = nullptr;
  std::map<int, int> prios;  // kernel nodes by the priority the capture recorded
};
// work captured from stream `st` (forks inside it join back before the end), run with each node's captured priority
static Graph capture(cudaStream_t st, const std::function<void()>& work) {
  cudaGraph_t g;
  Graph G;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  work();
  CK(cudaStreamEndCapture(st, &g));
  size_t n = 0;
  CK(cudaGraphGetNodes(g, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  CK(cudaGraphGetNodes(g, nodes.data(), &n));
  for (cudaGraphNode_t nd : nodes) {
    cudaGraphNodeType ty;
    CK(cudaGraphNodeGetType(nd, &ty));
    if (ty != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeAttrValue v{};
    if (cudaGraphKernelNodeGetAttribute(nd, cudaKernelNodeAttributePriority, &v) == cudaSuccess) G.prios[v.priority]++;
  }
  CK(cudaGraphInstantiateWithFlags(&G.ge, g, cudaGraphInstantiateFlagUseNodePriority));
  CK(cudaGraphDestroy(g));
  return G;
}
static void print_prios(const Graph& G) {
  printf("{");
  bool first = true;
  for (const auto& kv : G.prios) {
    printf("%s\"%d\": %d", first ? "" : ", ", kv.first, kv.second);
    first = false;
  }
  printf("}");
}
// reps of the graph on `st`, each timed by events, with the read SM clock and power
static std::vector<Rep> time_graph(Dev& d, const Graph& G, cudaStream_t st, int reps) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int i = 0; i < 2; i++) CK(cudaGraphLaunch(G.ge, st));
  CK(cudaStreamSynchronize(st));
  std::vector<Rep> out;
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(e0, st));
    CK(cudaGraphLaunch(G.ge, st));
    CK(cudaEventRecord(e1, st));
    double cs = 0, ps = 0;
    int nc = 0, np = 0;
    while (cudaEventQuery(e1) == cudaErrorNotReady) {
      const double c = nvml_mhz(d), p = nvml_w(d);
      if (c > 0) { cs += c; nc++; }
      if (p > 0) { ps += p; np++; }
    }
    CK(cudaEventSynchronize(e1));
    if (nc == 0) { cs = nvml_mhz(d); nc = 1; }
    if (np == 0) { ps = nvml_w(d); np = 1; }
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    out.push_back({ms, cs / nc, ps / np});
  }
  CK(cudaGetLastError());
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return out;
}
static double mean_ms(const std::vector<Rep>& v) {
  double s = 0;
  for (const Rep& r : v) s += r.ms;
  return v.empty() ? 0 : s / v.size();
}
static void print_reps(const char* name, const std::vector<Rep>& v, double macs, int sms, bool comma) {
  const Rep m = median(v);
  const double peak = 1024.0 * sms * m.mhz * 1e6;
  printf("\"%s\": {\"ms\": %.4f, \"mean_ms\": %.4f, \"mhz\": %.1f, \"watt\": %.1f, \"share_of_peak\": %.5f, \"reps_ms\": [", name,
         m.ms, mean_ms(v), m.mhz, m.watt, macs > 0 ? macs / (m.ms * 1e-3) / peak : 0.0);
  for (size_t i = 0; i < v.size(); i++) printf("%s%.4f", i ? ", " : "", v[i].ms);
  printf("]}%s", comma ? ", " : "");
}
static cudaStream_t prio_stream(int prio) {
  cudaStream_t s;
  CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, prio));
  return s;
}
static void fork(cudaStream_t from, cudaStream_t to) {
  cudaEvent_t e;
  CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  CK(cudaEventRecord(e, from));
  CK(cudaStreamWaitEvent(to, e, 0));
}
// gc_split's two partitions with C streams on each (stream c at priority prio + c), and n events made in each side's own
// context (an event is recorded only on a stream of its context; any stream may wait on it); 1000 x the failing stage +
// its error on failure
struct Split {
  int got[2] = {0, 0};
  std::vector<cudaStream_t> s[2];
  std::vector<cudaEvent_t> ev[2];
};
static int gc_split_n(int n_step, int C, int prio, int n, Split& g) {
  auto dget = drv<decltype(&cuDeviceGet)>("cuDeviceGet");
  auto res = drv<decltype(&cuDeviceGetDevResource)>("cuDeviceGetDevResource");
  auto split = drv<decltype(&cuDevSmResourceSplitByCount)>("cuDevSmResourceSplitByCount");
  auto gen = drv<decltype(&cuDevResourceGenerateDesc)>("cuDevResourceGenerateDesc");
  auto mk = drv<decltype(&cuGreenCtxCreate)>("cuGreenCtxCreate");
  auto mks = drv<decltype(&cuGreenCtxStreamCreate)>("cuGreenCtxStreamCreate");
  auto conv = drv<decltype(&cuCtxFromGreenCtx)>("cuCtxFromGreenCtx");
  auto push = drv<decltype(&cuCtxPushCurrent)>("cuCtxPushCurrent_v2");
  auto pop = drv<decltype(&cuCtxPopCurrent)>("cuCtxPopCurrent_v2");
  if (!dget || !res || !split || !gen || !mk || !mks || !conv || !push || !pop) return 1000;
  CUdevice dev;
  CUdevResource all{}, part[2]{};
  unsigned nb = 1;
  int e;
  if ((e = dget(&dev, 0))) return 2000 + e;
  if ((e = res(dev, &all, CU_DEV_RESOURCE_TYPE_SM))) return 2000 + e;
  if ((e = split(&part[0], &nb, &all, &part[1], 0, (unsigned)n_step))) return 3000 + e;
  for (int i = 0; i < 2; i++) {
    g.got[i] = (int)part[i].sm.smCount;
    CUdevResourceDesc desc;
    CUgreenCtx gc;
    CUcontext cx;
    if ((e = gen(&desc, &part[i], 1))) return 4000 + e;
    if ((e = mk(&gc, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM))) return 5000 + e;
    for (int c = 0; c < C; c++) {
      CUstream s;
      if ((e = mks(&s, gc, CU_STREAM_NON_BLOCKING, prio + c))) return 6000 + e;
      g.s[i].push_back((cudaStream_t)s);
    }
    if ((e = conv(&cx, gc))) return 7000 + e;
    if ((e = push(cx))) return 7000 + e;
    int ee = 0;
    for (int k = 0; k < n && !ee; k++) {
      cudaEvent_t ev;
      if (!(ee = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming))) g.ev[i].push_back(ev);
    }
    if ((e = pop(&cx))) return 7000 + e;
    if (ee) return 8000 + ee;
  }
  return 0;
}

// ---------------------------------------------------------------- prefill: the six stages of 32 layers of one chunk
constexpr int NSTAGE = 6;
static const int SK[NSTAGE] = {HID, QKV, 256, HID, HID, FF};  // the bytes a token of each stage's committed operand
static const char* SNAME[NSTAGE] = {"qkv", "qk", "pv", "o", "gate_up", "down"};
struct Chunk {
  uint8_t* op[NSTAGE];  // each stage's operand rows, M x SK
  int8_t* A;            // the GEMMs' A, M x FF
  int* D;               // their output
  Commit cm[NSTAGE];
  LtCtx lt;
};
struct Prefill {
  int M;
  LtB g[NSTAGE];
  const int8_t* W[LAYERS][NSTAGE];
  int gk[NSTAGE];
};
static void stage_gemm(const Prefill& P, Chunk& c, int l, int t, cudaStream_t st) {
  ltb_run(P.g[t], c.lt, c.A, P.W[l][t], c.D, st);
}
static int run_prefill(Dev& d, const std::string& dir, int M, int reps) {
  const int draw_us = getenv(RUN_DRAW_US) ? atoi(getenv(RUN_DRAW_US)) : 300;
  std::vector<HostConst> hc;
  for (int t = 0; t < NSTAGE; t++) hc.push_back(load_const(dir, SK[t], M));
  Prefill P;
  P.M = M;
  const int GN[NSTAGE] = {QKV, M / 2, M / 2, HID, 2 * FF, HID}, GK[NSTAGE] = {HID, HID, HID, HID, HID, FF};
  for (int l = 0; l < LAYERS; l++) {
    const int kv = M / 2 > HID ? 2 : 1;  // the attention stand-ins read (M/2) x HID bytes: o's weights up to M = 8192
    const int8_t* w[NSTAGE] = {d.W[l][0], d.W[l][kv], d.W[(l + 1) % LAYERS][kv], d.W[l][1], d.W[l][2], d.W[l][3]};
    for (int t = 0; t < NSTAGE; t++) P.W[l][t] = w[t];
  }
  const int NC = 3;
  std::vector<Chunk> ch(NC);
  for (int c = 0; c < NC; c++) {
    for (int t = 0; t < NSTAGE; t++) {
      CK(cudaMalloc(&ch[c].op[t], (size_t)M * SK[t]));
      fill_s8<<<1024, 256, 0, d.st>>>((int8_t*)ch[c].op[t], (size_t)M * SK[t], 0x4000 + 16 * c + t);
      ch[c].cm[t] = make_commit(hc[t]);
    }
    CK(cudaMalloc(&ch[c].A, (size_t)M * FF));
    fill_s8<<<1024, 256, 0, d.st>>>(ch[c].A, (size_t)M * FF, 0x5000 + c);
    CK(cudaMalloc(&ch[c].D, (size_t)M * 2 * FF * sizeof(int)));
    ch[c].lt = lt_ctx();
  }
  CK(cudaStreamSynchronize(d.st));
  for (int t = 0; t < NSTAGE; t++) {
    P.gk[t] = GK[t];
    P.g[t] = plan(M, GN[t], GK[t], 1, 0, 0, 0, ch[0].A, P.W[0][t], ch[0].D, d.st);
  }
  double macs = 0;
  for (int t = 0; t < NSTAGE; t++) macs += (double)LAYERS * M * GN[t] * GK[t];
  int lo, hi;  // least and greatest priority; a lower number is a higher priority
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t s0 = prio_stream(lo), s1 = prio_stream(lo), sm = prio_stream(lo);
  std::vector<cudaStream_t> cs;
  for (int c = 0; c < NC; c++) cs.push_back(prio_stream(std::min(hi + c, lo)));
  auto gemms = [&](Chunk& c, cudaStream_t st) {
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++) stage_gemm(P, c, l, t, st);
  };
  // the fast path's salt-only parts of a chunk's forward, a launch a stage for its 32 layers (salt), and a stage's
  // commitment after it (commit); hm96_rows.cu's commitment has no such part
  auto salt = [&](Chunk& c, int t, cudaStream_t st) { if (g_fast) commit_salt(c.cm[t], LAYERS, st); };
  auto commit = [&](Chunk& c, int l, int t, cudaStream_t st) {
    if (g_fast) commit_tree(c.cm[t], l, c.op[t], st);
    else commit_run(c.cm[t], c.op[t], st);
  };
  auto hashes = [&](Chunk& c, cudaStream_t st) {
    for (int t = 0; t < NSTAGE; t++) salt(c, t, st);
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++) commit(c, l, t, st);
  };
  // ahead: the salt-only parts on a side stream at the start, each stage's first commitment waiting for its own, so they
  // leave the critical path; otherwise inline at the start
  std::vector<cudaStream_t> side;
  for (int c = 0; c < NC; c++) side.push_back(prio_stream(std::min(hi + c, lo)));
  auto stage_seq = [&](Chunk& c, cudaStream_t st, bool hash, bool draw, cudaStream_t ahead) {
    std::vector<cudaEvent_t> ev;
    if (hash && g_fast) {
      if (ahead) fork(st, ahead);
      for (int t = 0; t < NSTAGE; t++) {
        salt(c, t, ahead ? ahead : st);
        cudaEvent_t e;
        CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        CK(cudaEventRecord(e, ahead ? ahead : st));
        ev.push_back(e);
      }
    }
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++) {
        if (hash) {
          if (l == 0 && !ev.empty()) CK(cudaStreamWaitEvent(st, ev[t], 0));
          commit(c, l, t, st);
        }
        if (draw) draw_wait<<<1, 1, 0, st>>>((unsigned long long)draw_us * 1000ull);
        stage_gemm(P, c, l, t, st);
      }
  };
  std::vector<std::pair<std::string, std::vector<Rep>>> arms;
  std::vector<double> arm_macs;
  std::vector<Graph> arm_graphs;
  auto add = [&](const std::string& nm, cudaStream_t st, const std::function<void()>& w, double mm) {
    const Graph G = capture(st, w);
    arms.push_back({nm, time_graph(d, G, st, reps)});
    arm_macs.push_back(mm);
    CK(cudaGraphExecDestroy(G.ge));
    arm_graphs.push_back(G);
  };
  const char* arms_env = getenv("ARMS");  // r1 (round 1's arms), r2 (the proved forward's), both by default
  const bool r1 = !arms_env || strstr(arms_env, "r1"), r2 = !arms_env || strstr(arms_env, "r2");
  add("gemm", sm, [&] { gemms(ch[0], sm); }, macs);
  if (r1) add("hash", sm, [&] { hashes(ch[0], sm); }, 0);
  if (r1 && g_fast) {
    add("salt", sm, [&] { for (int t = 0; t < NSTAGE; t++) salt(ch[0], t, sm); }, 0);
    add("tree", sm, [&] {
      for (int l = 0; l < LAYERS; l++)
        for (int t = 0; t < NSTAGE; t++) commit_tree(ch[0].cm[t], l, ch[0].op[t], sm);
    }, 0);
  }
  cudaStream_t s1lo = prio_stream(lo), s0hi = prio_stream(hi);
  if (r1) {
    add("serial", sm, [&] { stage_seq(ch[0], sm, true, false, nullptr); }, macs);
    add("conc", sm, [&] {
      fork(sm, s0);
      gemms(ch[0], s0);
      fork(sm, s1);
      hashes(ch[1], s1);
      fork(s0, sm);
      fork(s1, sm);
    }, macs);
    add("conc_lo", sm, [&] {
      fork(sm, s0hi);
      gemms(ch[0], s0hi);
      fork(sm, s1lo);
      hashes(ch[1], s1lo);
      fork(s0hi, sm);
      fork(s1lo, sm);
    }, macs);
  }
  // hash latency a stage: 32 commitments of one stage's operands back to back (the fast path's critical part: its salt-only
  // part, timed apart, runs ahead)
  std::vector<std::pair<std::string, std::vector<Rep>>> lat, salt_lat, seg_lat, tree_lat;
  for (int t = 0; t < NSTAGE; t++) {
    const Graph G = capture(sm, [&] {
      for (int l = 0; l < LAYERS; l++)
        if (g_fast) commit_tree(ch[0].cm[t], l, ch[0].op[t], sm);
        else commit_run(ch[0].cm[t], ch[0].op[t], sm);
    });
    lat.push_back({SNAME[t], time_graph(d, G, sm, reps)});
    CK(cudaGraphExecDestroy(G.ge));
    if (!g_fast) continue;
    const Graph S = capture(sm, [&] { salt(ch[0], t, sm); });
    salt_lat.push_back({SNAME[t], time_graph(d, S, sm, reps)});
    CK(cudaGraphExecDestroy(S.ge));
    const Graph Sg = capture(sm, [&] { for (int l = 0; l < LAYERS; l++) commit_segs(ch[0].cm[t], ch[0].op[t], sm); });
    seg_lat.push_back({SNAME[t], time_graph(d, Sg, sm, reps)});
    CK(cudaGraphExecDestroy(Sg.ge));
    const Graph Tr = capture(sm, [&] { for (int l = 0; l < LAYERS; l++) commit_frtree(ch[0].cm[t], l, sm); });
    tree_lat.push_back({SNAME[t], time_graph(d, Tr, sm, reps)});
    CK(cudaGraphExecDestroy(Tr.ge));
  }
  // the phase probe: one commitment of qkv's and of down's operands after warm ones, every tree CTA's phase clocks (fr_tp)
  // from the segments' first start
  if (g_fast) {
    const int S = FR_TP_SLOTS, maxc = 1024;
    unsigned long long *tp, *tps, *null = nullptr;
    CK(cudaMalloc(&tp, sizeof(unsigned long long) * maxc * S));
    CK(cudaMalloc(&tps, 2 * sizeof(unsigned long long)));
    for (int t : {0, 5}) {
      const Commit& m = ch[0].cm[t];
      commit_salt(m, LAYERS, sm);
      for (int r = 0; r < 3; r++) commit_tree(m, 0, ch[0].op[t], sm);
      CK(cudaStreamSynchronize(sm));
      CK(cudaMemset(tp, 0, sizeof(unsigned long long) * maxc * S));
      const unsigned long long init[2] = {~0ull, 0};
      CK(cudaMemcpy(tps, init, sizeof init, cudaMemcpyHostToDevice));
      CK(cudaMemcpyToSymbol(fr_tp, &tp, sizeof tp));
      CK(cudaMemcpyToSymbol(fr_tps, &tps, sizeof tps));
      commit_tree(m, 0, ch[0].op[t], sm);
      CK(cudaStreamSynchronize(sm));
      CK(cudaMemcpyToSymbol(fr_tp, &null, sizeof null));
      CK(cudaMemcpyToSymbol(fr_tps, &null, sizeof null));
      std::vector<unsigned long long> h((size_t)maxc * S), hs(2);
      CK(cudaMemcpy(h.data(), tp, h.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hs.data(), tps, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      const int total = m.width / std::min(m.width, 128);
      const unsigned long long t0 = hs[0];
      auto us = [&](unsigned long long x) { return x ? (double)(long long)(x - t0) / 1e3 : -1.0; };
      int lastc = -1;
      unsigned long long first = ~0ull;
      std::vector<double> sub;
      for (int c = 0; c < total; c++) {
        if (h[(size_t)c * S + 13]) lastc = c;
        first = std::min(first, h[(size_t)c * S]);
        unsigned long long dn = 0;
        for (int k = 4; k <= 12; k++) dn = std::max(dn, h[(size_t)c * S + k]);
        sub.push_back(us(dn));
      }
      std::sort(sub.begin(), sub.end());
      printf("{\"mode\": \"phases\", \"M\": %d, \"stage\": \"%s\", \"ctas\": %d, \"seg_end_us\": %.2f, \"tree_start_us\": %.2f, "
             "\"subtree_done_us\": [%.2f, %.2f, %.2f], \"cta0_us\": [",
             M, SNAME[t], total, us(hs[1]), us(first), sub.front(), sub[sub.size() / 2], sub.back());
      for (int k = 0; k < S; k++) printf("%s%.2f", k ? ", " : "", us(h[k]));
      printf("], \"last_cta\": %d, \"last_us\": [", lastc);
      for (int k = 0; k < S; k++) printf("%s%.2f", k ? ", " : "", lastc >= 0 ? us(h[(size_t)lastc * S + k]) : -1.0);
      printf("]}\n");
      fflush(stdout);
    }
    CK(cudaFree(tp));
    CK(cudaFree(tps));
  }
  // the proved forward, lockstep: a stage's operands committed, the draw, then its GEMM, stage by stage. k > 1: every GEMM
  // in k row slabs, the next stage's operands hashed slab by slab behind it on h (the producing kernel's epilogue, emulated:
  // cuBLASLt's can't be fused), so at its end only the last slab's leaves and the top are left; the first stage's operands
  // (the embedding's) whole. draw 0: none; 1: DRAW_US on the GPU; 2: the host's (the tree stops at E nodes for the host to
  // hash the top, E = 1 the GPU's root; then the host waits its draw, dus, and raises the GEMM's flag). The salt-only parts
  // and first-block states ahead on a side stream.
  // P: the host's threads for the top (h_top's pool); col: the k slabs are column slabs, each GEMM slab followed by the
  // segment pass over the next operand's blocks it wrote (fr_segcols, in order on h), the leaves and tree after the last
  // nopdl: fr_tree launched after its segment pass completes, not programmatically (g_pdl)
  struct Cfg {
    std::string name;
    int mid, k, E, draw, dus, pair, P, col, nopdl;
  };
  std::vector<Cfg> cfgs;
  std::vector<int> ks, kc;  // row and column slab counts
  if (r2) {
    const int du = draw_us;
    const char* sel = getenv("CFGS");  // a comma-separated subset by name (all by default)
    std::vector<Cfg> all = {{"mid", 1, 1, 0, 1, du, 0, 1, 0},
                            {"pair", 1, 1, 0, 1, du, 1, 1, 0},
                            {"cpu64", 1, 1, 64, 2, du, 0, 1, 0},
                            {"cpu256p8", 1, 1, 256, 2, du, 0, 8, 0},
                            {"cpu512p16", 1, 1, 512, 2, du, 0, 16, 0},
                            {"cpu1024p32", 1, 1, 1024, 2, du, 0, 32, 0},
                            {"pair_cpu512p16", 1, 1, 512, 2, du, 1, 16, 0},
                            {"s4", 1, 4, 0, 1, du, 0, 1, 0},
                            {"s4_cpu512p16", 1, 4, 512, 2, du, 0, 16, 0},
                            {"c8", 1, 8, 0, 1, du, 0, 1, 1},
                            {"c8_cpu512p16", 1, 8, 512, 2, du, 0, 16, 1},
                            {"c16_cpu512p16", 1, 16, 512, 2, du, 0, 16, 1},
                            {"pair_cpu256p8", 1, 1, 256, 2, du, 1, 8, 0},
                            {"c8_pair_cpu512p16", 1, 8, 512, 2, du, 1, 16, 1},
                            {"mid_nd", 1, 1, 0, 0, 0, 0, 1, 0},
                            {"pair_nd", 1, 1, 0, 0, 0, 1, 1, 0},
                            {"cpu512p16_nd", 1, 1, 512, 2, 0, 0, 16, 0},
                            {"cpu1024p32_nd", 1, 1, 1024, 2, 0, 0, 32, 0},
                            {"pair_cpu1024p32_nd", 1, 1, 1024, 2, 0, 1, 32, 0},
                            {"s4_cpu512p16_nd", 1, 4, 512, 2, 0, 0, 16, 0},
                            {"c8_cpu512p16_nd", 1, 8, 512, 2, 0, 0, 16, 1},
                            {"c16_cpu512p16_nd", 1, 16, 512, 2, 0, 0, 16, 1},
                            {"c8_pair_cpu512p16_nd", 1, 8, 512, 2, 0, 1, 16, 1},
                            {"pair_cpu1024p32", 1, 1, 1024, 2, du, 1, 32, 0},
                            {"pair_cpu512p16_nd", 1, 1, 512, 2, 0, 1, 16, 0},
                            {"cpu256p8_nopdl", 1, 1, 256, 2, du, 0, 8, 0, 1},
                            {"cpu512p16_nd_nopdl", 1, 1, 512, 2, 0, 0, 16, 0, 1}};
    for (const Cfg& f : all)
      if (!sel || strstr(("," + std::string(sel) + ",").c_str(), ("," + f.name + ",").c_str())) {
        if (f.k > 1 && !f.col && M % (128 * f.k)) {
          fprintf(stderr, "prefill M=%d: %s skipped: its row slabs aren't whole tree CTAs of 128 rows, and each slab's hash "
                  "stream waits on its own GEMM slab alone\n", M, f.name.c_str());
          continue;
        }
        cfgs.push_back(f);
      }
    for (const Cfg& f : cfgs) {
      if (f.k > 1 && !f.col && std::find(ks.begin(), ks.end(), f.k) == ks.end()) ks.push_back(f.k);
      if (f.col && std::find(kc.begin(), kc.end(), f.k) == kc.end()) kc.push_back(f.k);
    }
  }
  for (int k : ks)
    for (int t = 0; t < NSTAGE; t++) plan(M / k, GN[t], GK[t], 1, 0, 0, 0, ch[0].A, P.W[0][t], ch[0].D, d.st);
  for (int k : kc)
    for (int t = 0; t < NSTAGE; t++) plan(M, GN[t] / k, GK[t], 1, 0, 0, 0, ch[0].A, P.W[0][t], ch[0].D, d.st);
  // column slab i of k: weight rows [i N/k, (i+1) N/k), its output its own M x N/k block of D (the timing's, not a layout's)
  auto col_gemm = [&](Chunk& c, int l, int t, int k, int i, cudaStream_t st) {
    const LtB& B = g_plans.at(std::make_tuple(M, GN[t] / k, GK[t], 1));
    ltb_run(B, c.lt, c.A, P.W[l][t] + (long long)i * (GN[t] / k) * GK[t], c.D + (long long)i * M * (GN[t] / k), st);
  };
  auto slab_gemm = [&](Chunk& c, int l, int t, int k, int i, cudaStream_t st) {
    const LtB& B = g_plans.at(std::make_tuple(M / k, GN[t], GK[t], 1));
    const long long r0 = (long long)i * (M / k);
    ltb_run(B, c.lt, c.A + r0 * GK[t], P.W[l][t], c.D + r0 * GN[t], st);
  };
  cudaStream_t hs = prio_stream(hi), as = prio_stream(lo);
  std::vector<cudaStream_t> hsl;  // a hash stream a slab, so the slabs' latency-bound chains overlap
  for (int i = 0; i < 8; i++) hsl.push_back(prio_stream(hi));
  auto proved = [&](Chunk& c, cudaStream_t g, cudaStream_t h, cudaStream_t ahead, const Cfg& f) {
    std::vector<cudaEvent_t> ev;
    fork(g, ahead);
    for (int t = 0; t < NSTAGE; t++) {
      salt(c, t, ahead);
      cudaEvent_t e;
      CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      CK(cudaEventRecord(e, ahead));
      ev.push_back(e);
    }
    const int Q = LAYERS * NSTAGE;
    for (int q = 0; q < Q; q++) {
      const int l = q / NSTAGE, t = q % NSTAGE;
      const Export x{g_link.ex_d, g_link.up_d, f.E, (uint32_t)q + 1};
      if (f.k == 1 || q == 0) {
        fork(g, h);
        if (q < NSTAGE) CK(cudaStreamWaitEvent(h, ev[t], 0));
        commit_tree(c.cm[t], l, c.op[t], h, f.draw == 2 ? &x : nullptr);
      }
      if (f.draw == 1) draw_wait<<<1, 1, 0, h>>>((unsigned long long)f.dus * 1000ull);
      if (f.draw == 2) host_wait<<<1, 1, 0, g>>>(g_link.dn_d, (uint32_t)q + 1, g_link.err_d);
      else fork(h, g);
      if (f.k == 1 || q + 1 == Q) {
        stage_gemm(P, c, l, t, g);
        continue;
      }
      const int qn = q + 1, ln = qn / NSTAGE, tn = qn % NSTAGE;
      const Commit& m = c.cm[tn];
      const Export xn{g_link.ex_d, g_link.up_d, f.E, (uint32_t)qn + 1};
      if (f.col) {
        const int B = row_blocks(m);
        if (qn < NSTAGE) CK(cudaStreamWaitEvent(h, ev[tn], 0));
        for (int i = 0; i < f.k; i++) {
          col_gemm(c, l, t, f.k, i, g);
          fork(g, h);
          commit_segcols(m, c.op[tn], h, i * B / f.k, (i + 1) * B / f.k);
        }
        commit_frtree(m, ln, h, 0, -1, f.draw == 2 ? &xn : nullptr);
        continue;
      }
      for (int i = 0; i < f.k; i++) {
        slab_gemm(c, l, t, f.k, i, g);
        fork(g, hsl[i]);
        if (qn < NSTAGE) CK(cudaStreamWaitEvent(hsl[i], ev[tn], 0));
        commit_slab(m, ln, c.op[tn], hsl[i], f.k, i, f.draw == 2 ? &xn : nullptr);
        if (f.draw != 2) fork(hsl[i], h);
      }
    }
    fork(h, g);
    for (int i = 0; f.k > 1 && !f.col && i < f.k; i++) fork(hsl[i], g);
    fork(ahead, g);
  };
  struct HostArm {
    std::string name;
    HostStats st;
    uint32_t gpu_timeouts;
  };
  std::vector<HostArm> host_arms;
  for (int k : ks) add("gemm_s" + std::to_string(k), sm, [&] {
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++)
        for (int i = 0; i < k; i++) slab_gemm(ch[0], l, t, k, i, sm);
  }, macs);
  for (int k : kc) add("gemm_c" + std::to_string(k), sm, [&] {
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++)
        for (int i = 0; i < k; i++) col_gemm(ch[0], l, t, k, i, sm);
  }, macs);
  for (const Cfg& f : cfgs) {
    g_mid = f.mid;
    g_pair = f.pair;
    g_pdl = !f.nopdl;
    const Graph G = capture(sm, [&] { proved(ch[0], sm, hs, as, f); });
    g_mid = 1;
    g_pair = 0;
    g_pdl = 1;
    HostStats st;
    std::thread th;
    if (f.draw == 2) {
      std::vector<HostStep> steps;
      for (int q = 0; q < LAYERS * NSTAGE; q++) steps.push_back({&hc[q % NSTAGE], f.E});
      for (int t = 0; t < NSTAGE; t++) h_nmid(hc[t]);
      h_store(g_link.up_h, 0);
      h_store(g_link.dn_h, 0);
      CK(cudaMemset(g_link.err_d, 0, 4));
      if (f.P > 1) pool_start(f.P);
      th = std::thread(host_loop, steps, reps + 2, f.dus, &st);
    }
    arms.push_back({f.name, time_graph(d, G, sm, reps)});
    arm_macs.push_back(macs);
    arm_graphs.push_back(G);
    if (th.joinable()) {
      th.join();
      uint32_t err = 0;
      CK(cudaMemcpy(&err, g_link.err_d, 4, cudaMemcpyDeviceToHost));
      host_arms.push_back({f.name, st, err});
      if (f.P > 1) pool_stop();
    }
    CK(cudaGraphExecDestroy(G.ge));
  }
  // C chunks in flight, each hash -> draw -> GEMM a stage, chunk 0 at the highest priority
  for (int C = 1; C <= NC && r1; C++) {
    add("pipe" + std::to_string(C), sm, [&] {
      for (int c = 0; c < C; c++) {
        fork(sm, cs[c]);
        stage_seq(ch[c], cs[c], true, true, side[c]);
      }
      for (int c = 0; c < C; c++) fork(cs[c], sm);
    }, C * macs);
  }
  if (r1) {
    add("pipe1_draw", sm, [&] { stage_seq(ch[0], sm, false, true, nullptr); }, macs);
    add("pipe1_hash", sm, [&] { stage_seq(ch[0], sm, true, false, side[0]); }, macs);
  }
  // the commitments on a green partition of N SMs, the GEMMs on the rest (no graph: the two contexts' streams)
  std::vector<std::tuple<int, int, int, double, double, double>> green;  // N asked, gemm SMs, hash SMs, gemm alone, both, hash alone
  for (int N : {28, 44}) {
    if (!r1) break;
    cudaStream_t sg = nullptr, sh = nullptr;
    int got[2] = {0, 0};
    const int e = gc_split(d.sms - N, &sg, &sh, got);
    if (e || !sh) { fprintf(stderr, "gc_split(%d): %d\n", d.sms - N, e); continue; }
    auto host_ms = [&](bool g, bool h) {
      std::vector<double> v;
      for (int r = 0; r < reps + 1; r++) {
        CK(cudaDeviceSynchronize());
        const double t0 = now_ms();
        if (g) gemms(ch[0], sg);
        if (h) hashes(ch[1], sh);
        CK(cudaStreamSynchronize(sg));
        CK(cudaStreamSynchronize(sh));
        const double t1 = now_ms();
        if (r) v.push_back(t1 - t0);
      }
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    const double ga = host_ms(true, false), both = host_ms(true, true), ha = host_ms(false, true);
    green.emplace_back(N, got[0], got[1], ga, both, ha);
  }
  // C chunks in flight on a split: each stage's commitment and draw on the hash partition's stream c, its GEMM on the
  // rest's, the commitment after the stage before's GEMM (events; host-launched, against C x gemm)
  std::vector<std::tuple<int, int, int, int, double, double>> pgreen;  // N asked, C, gemm SMs, hash SMs, GEMMs alone, pipelined
  const bool pg_on = !getenv("PIPE_GREEN") || atoi(getenv("PIPE_GREEN"));  // PIPE_GREEN=0 skips these arms
  for (int N : {20, 28, 36, 44}) {
    if (!pg_on || !r1) break;
    Split g;
    int e = gc_split_n(d.sms - N, NC, hi, NC * LAYERS * NSTAGE, g);
    if (e) {
      fprintf(stderr, "gc_split_n(%d): %d\n", d.sms - N, e);
      cudaGetLastError();
      continue;
    }
    auto ev = [&](int side, int c, int l, int t) { return g.ev[side][(c * LAYERS + l) * NSTAGE + t]; };
    for (int C = 2; C <= NC; C++) {
      auto run = [&](bool hash) -> int {
        int err;
        if (hash)
          for (int c = 0; c < C; c++)
            for (int t = 0; t < NSTAGE; t++) salt(ch[c], t, g.s[1][c]);
        for (int l = 0; l < LAYERS; l++)
          for (int t = 0; t < NSTAGE; t++)
            for (int c = 0; c < C; c++) {
              cudaStream_t sh = g.s[1][c], sg = g.s[0][c];
              if (hash) {
                const int pl = t ? l : l - 1, pt = t ? t - 1 : NSTAGE - 1;
                if (pl >= 0 && (err = cudaStreamWaitEvent(sh, ev(0, c, pl, pt), 0))) return err;
                commit(ch[c], l, t, sh);
                draw_wait<<<1, 1, 0, sh>>>((unsigned long long)draw_us * 1000ull);
                if ((err = cudaEventRecord(ev(1, c, l, t), sh))) return err;
                if ((err = cudaStreamWaitEvent(sg, ev(1, c, l, t), 0))) return err;
              }
              stage_gemm(P, ch[c], l, t, sg);
              if (hash && (err = cudaEventRecord(ev(0, c, l, t), sg))) return err;
            }
        return 0;
      };
      auto host_ms = [&](bool hash, double* ms) -> int {
        std::vector<double> v;
        for (int r = 0; r < reps + 1; r++) {
          CK(cudaDeviceSynchronize());
          const double t0 = now_ms();
          if (const int err = run(hash)) return err;
          for (int c = 0; c < C; c++) {
            CK(cudaStreamSynchronize(g.s[0][c]));
            CK(cudaStreamSynchronize(g.s[1][c]));
          }
          if (r) v.push_back(now_ms() - t0);
        }
        std::sort(v.begin(), v.end());
        *ms = v[v.size() / 2];
        return 0;
      };
      double ga = 0, pp = 0;
      if ((e = host_ms(false, &ga)) || (e = host_ms(true, &pp))) {
        fprintf(stderr, "pipe_green N=%d C=%d: %d\n", N, C, e);
        cudaGetLastError();
        break;
      }
      pgreen.emplace_back(N, C, g.got[0], g.got[1], ga, pp);
    }
  }
  printf("{\"mode\": \"prefill\", \"hash\": \"%s\", \"M\": %d, \"layers\": %d, \"draw_us\": %d, \"chunks_max\": %d, \"macs\": %.6e, "
         "\"op_bytes_token_layer\": %d, \"rejected\": %d, \"missing_algo\": %d, \"gemms\": [",
         g_fast ? "fr" : "sz", M, LAYERS, draw_us, NC, macs, SK[0] + SK[1] + SK[2] + SK[3] + SK[4] + SK[5], g_rejected_total,
         g_missing);
  for (int t = 0; t < NSTAGE; t++)
    printf("%s{\"stage\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"algo\": %d, \"ms_alone\": %.4f}", t ? ", " : "", SNAME[t], M,
           GN[t], GK[t], P.g[t].L.algo_id, P.g[t].L.ms);
  printf("], \"hash_latency\": {");
  for (size_t i = 0; i < lat.size(); i++) print_reps(lat[i].first.c_str(), lat[i].second, 0, d.sms, i + 1 < lat.size());
  printf("}, \"salt_latency\": {");
  for (size_t i = 0; i < salt_lat.size(); i++)
    print_reps(salt_lat[i].first.c_str(), salt_lat[i].second, 0, d.sms, i + 1 < salt_lat.size());
  printf("}, \"segments_latency\": {");
  for (size_t i = 0; i < seg_lat.size(); i++)
    print_reps(seg_lat[i].first.c_str(), seg_lat[i].second, 0, d.sms, i + 1 < seg_lat.size());
  printf("}, \"tree_latency\": {");
  for (size_t i = 0; i < tree_lat.size(); i++)
    print_reps(tree_lat[i].first.c_str(), tree_lat[i].second, 0, d.sms, i + 1 < tree_lat.size());
  printf("}, \"green\": [");
  for (size_t i = 0; i < green.size(); i++) {
    const auto& g = green[i];
    printf("%s{\"N\": %d, \"gemm_sms\": %d, \"hash_sms\": %d, \"gemm_ms\": %.4f, \"both_ms\": %.4f, \"hash_ms\": %.4f}", i ? ", " : "",
           std::get<0>(g), std::get<1>(g), std::get<2>(g), std::get<3>(g), std::get<4>(g), std::get<5>(g));
  }
  printf("], \"pipe_green\": [");
  for (size_t i = 0; i < pgreen.size(); i++) {
    const auto& g = pgreen[i];
    printf("%s{\"N\": %d, \"C\": %d, \"gemm_sms\": %d, \"hash_sms\": %d, \"gemm_ms\": %.4f, \"ms\": %.4f}", i ? ", " : "",
           std::get<0>(g), std::get<1>(g), std::get<2>(g), std::get<3>(g), std::get<4>(g), std::get<5>(g));
  }
  printf("], \"node_priorities\": {");
  for (size_t i = 0; i < arms.size(); i++) {
    printf("%s\"%s\": ", i ? ", " : "", arms[i].first.c_str());
    print_prios(arm_graphs[i]);
  }
  printf("}, \"host_sha\": {\"uses\": \"%s\", \"cpp_us\": %.4f, \"ossl_us\": %.4f}, \"host\": {", g_use_ossl ? "ossl" : "cpp",
         g_cpp_us, g_ossl_us);
  int bad = 0;
  for (size_t i = 0; i < host_arms.size(); i++) {
    const HostArm& h = host_arms[i];
    printf("%s\"%s\": {\"steps\": %ld, \"top_us_mean\": %.3f, \"top_us_max\": %.3f, \"host_timeouts\": %ld, \"gpu_timeouts\": %u}",
           i ? ", " : "", h.name.c_str(), h.st.steps, h.st.steps ? h.st.top_us / h.st.steps : 0.0, h.st.top_max_us,
           h.st.timeouts, h.gpu_timeouts);
    bad += h.st.timeouts > 0 || h.gpu_timeouts > 0 || h.st.steps != (long)(reps + 2) * LAYERS * NSTAGE;
  }
  printf("}, \"proved_arms\": [");
  for (size_t i = 0; i < cfgs.size(); i++)
    printf("%s{\"name\": \"%s\", \"k\": %d, \"E\": %d, \"draw\": %d, \"P\": %d, \"pair\": %d, \"col\": %d, \"pdl\": %d}",
           i ? ", " : "", cfgs[i].name.c_str(), cfgs[i].k, cfgs[i].E, cfgs[i].draw, cfgs[i].P, cfgs[i].pair, cfgs[i].col,
           !cfgs[i].nopdl);
  printf("], \"arms\": {");
  for (size_t i = 0; i < arms.size(); i++) print_reps(arms[i].first.c_str(), arms[i].second, arm_macs[i], d.sms, i + 1 < arms.size());
  printf("}}\n");
  fflush(stdout);
  return bad ? 7 : 0;
}

// ---------------------------------------------------------------- the deferred pass: one layer's noised products over T
struct Seq {
  int t, ctx, lk;
  int8_t *K, *V;  // [HKV][lk][HD] and [HKV][HD][lk]
};
static bool g_att_fused = true;  // ATT=lt: attention as cuBLASLt's batched GEMMs with S and P in memory
struct Pass {
  int T;
  bool b32;
  std::vector<Seq> seqs;
  int8_t *X, *Q, *Qn, *P, *Wf, *Xf, *KVf;
  int *D, *S, *O, *csum;
  size_t lin_rows_max;
  double lin_macs = 0, att_macs = 0, form_bytes = 0, stored_bytes_token_layer = 0;
  LtCtx lt;
};
static int ctx_of(int i, int n) {
  if (n <= 1) return 4608;
  return 1024 + (int)std::lround((8192.0 - 1024.0) * i / (n - 1) / 512.0) * 512;
}
static Pass make_pass(Dev& d, int T, bool b32) {
  Pass p;
  p.T = T;
  p.b32 = b32;
  const int nseq = b32 ? 32 : std::max(1, T / 512), t = T / nseq;
  size_t smax = 0, kvmax = 0;
  for (int i = 0; i < nseq; i++) {
    Seq s;
    s.t = t;
    s.ctx = ctx_of(i, nseq);
    s.lk = (s.ctx + t / 2 + 31) / 32 * 32;
    const size_t kv = (size_t)HKV * s.lk * HD;
    CK(cudaMalloc(&s.K, kv));
    CK(cudaMalloc(&s.V, kv));
    fill_s8<<<1024, 256, 0, d.st>>>(s.K, kv, 0x6000 + 2 * i);
    fill_s8<<<1024, 256, 0, d.st>>>(s.V, kv, 0x6001 + 2 * i);
    smax = std::max(smax, (size_t)HKV * 4 * t * s.lk);
    kvmax = std::max(kvmax, kv);
    p.seqs.push_back(s);
    p.att_macs += 3.0 * HKV * 4.0 * t * s.lk * HD;
  }
  p.lin_rows_max = std::min(T, 8192);
  CK(cudaMalloc(&p.X, p.lin_rows_max * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(p.X, p.lin_rows_max * FF, 0x7000);
  CK(cudaMalloc(&p.Xf, p.lin_rows_max * FF));
  CK(cudaMalloc(&p.D, p.lin_rows_max * 2 * FF * sizeof(int)));
  CK(cudaMalloc(&p.Wf, (size_t)2 * FF * HID));
  CK(cudaMalloc(&p.KVf, kvmax));
  CK(cudaMalloc(&p.Q, (size_t)HKV * 4 * t * HD));
  fill_s8<<<1024, 256, 0, d.st>>>(p.Q, (size_t)HKV * 4 * t * HD, 0x7001);
  CK(cudaMalloc(&p.Qn, (size_t)HKV * 4 * t * HD));
  fill_s8<<<1024, 256, 0, d.st>>>(p.Qn, (size_t)HKV * 4 * t * HD, 0x7003);
  CK(cudaMalloc(&p.csum, (size_t)HKV * 4 * t * sizeof(int)));
  CK(cudaMalloc(&p.P, smax));
  fill_s8<<<1024, 256, 0, d.st>>>(p.P, smax, 0x7002);
  CK(cudaMalloc(&p.S, smax * sizeof(int)));
  CK(cudaMalloc(&p.O, (size_t)HKV * 4 * t * HD * sizeof(int)));
  for (int j = 0; j < 4; j++) p.lin_macs += (double)T * LN[j] * LK[j];
  // what the deferred edit keeps from the served pass, a token-layer: QKV's input, Q (K and V are the cache), attention's
  // output, gate_up's and down's inputs, and the row normalizers (m, Z) of the 32 query heads
  p.stored_bytes_token_layer = HID + HQ * HD + HID + HID + FF + HQ * 2 * 4;
  p.lt = lt_ctx();
  CK(cudaStreamSynchronize(d.st));
  return p;
}
// one layer of the pass on stream st; plans are made (and cached) on d.st beforehand by pass_plans
static void pass_layer(Dev& d, Pass& p, int layer, bool forming, cudaStream_t st, bool plan_only = false) {
  const int8_t* W[4] = {d.W[layer][0], d.W[layer][1], d.W[layer][2], d.W[layer][3]};
  p.form_bytes = 0;
  for (int j = 0; j < 4; j++) {
    if (forming && !plan_only) form(W[j], p.Wf, (size_t)LN[j] * LK[j], 0x100 + j, st, d.sms);
    p.form_bytes += 2.0 * LN[j] * LK[j];
    for (int r0 = 0; r0 < p.T; r0 += (int)p.lin_rows_max) {
      const int m = std::min((int)p.lin_rows_max, p.T - r0);
      if (forming && !plan_only) form(p.X, p.Xf, (size_t)m * LK[j], 0x200 + j, st, d.sms);
      p.form_bytes += 2.0 * m * LK[j];
      const LtB& B = plan(m, LN[j], LK[j], 1, 0, 0, 0, p.X, W[j], p.D, d.st);
      if (!plan_only) ltb_run(B, p.lt, forming ? p.Xf : p.X, forming ? p.Wf : W[j], p.D, st);
    }
  }
  for (const Seq& s : p.seqs) {
    const int m = 4 * s.t;
    const size_t kv = (size_t)HKV * s.lk * HD;
    if (g_att_fused && !plan_only) {
      if (forming) form(p.Q, p.Qn, (size_t)HKV * m * HD, 0x303, st, d.sms);
      att_run(p.Q, p.Qn, s.K, s.V, m, s.lk, p.O, p.csum, 0x304, st);
      continue;
    }
    const LtB& QK = plan(m, s.lk, HD, HKV, (long long)m * HD, (long long)s.lk * HD, (long long)m * s.lk, p.Q, s.K, p.S, d.st);
    const LtB& PV = plan(m, HD, s.lk, HKV, (long long)m * s.lk, (long long)HD * s.lk, (long long)m * HD, p.P, s.V, p.O, d.st);
    if (plan_only) continue;
    ltb_run(QK, p.lt, p.Q, s.K, p.S, st);  // clean, to rebuild P
    if (forming) {
      const size_t sn = (size_t)HKV * m * s.lk;
      soft_q<<<d.sms * 8, 256, 0, st>>>((const int4*)p.S, (int*)p.P, sn / 4);
      form(s.K, p.KVf, kv, 0x300, st, d.sms);
      form(s.V, p.KVf, kv, 0x301, st, d.sms);
      form(p.P, p.P, sn, 0x302, st, d.sms);
    }
    ltb_run(QK, p.lt, p.Q, forming ? p.KVf : s.K, p.S, st);
    ltb_run(PV, p.lt, p.P, forming ? p.KVf : s.V, p.O, st);
  }
  if (forming) {
    double att = 0;
    for (const Seq& s : p.seqs)
      att += g_att_fused ? 2.0 * HKV * 4 * s.t * HD : 4.0 * HKV * s.lk * HD + (double)HKV * 4 * s.t * s.lk * 7;
    p.form_bytes += att;
  }
}
static const int PASS_LAYERS = 4;
static void print_pass(const Pass& p) {
  int nseq = (int)p.seqs.size();
  double ctx = 0;
  for (const Seq& s : p.seqs) ctx += s.ctx;
  printf("\"T\": %d, \"comp\": \"%s\", \"att\": \"%s\", \"seqs\": %d, \"tokens_a_seq\": %d, \"ctx_mean\": %.0f, \"lin_macs\": %.6e, "
         "\"att_macs\": %.6e, \"stored_bytes_token_layer\": %.0f, \"stored_bytes_replica\": %.6e, \"form_bytes\": %.6e",
         p.T, p.b32 ? "b32" : "seq500", g_att_fused ? "fused" : "lt", nseq, p.seqs[0].t, ctx / nseq, p.lin_macs, p.att_macs,
         p.stored_bytes_token_layer, p.stored_bytes_token_layer * LAYERS * p.T, p.form_bytes);
}
static int run_deferred(Dev& d, int T, bool b32, int reps) {
  Pass p = make_pass(d, T, b32);
  pass_layer(d, p, 0, true, d.st, true);
  cudaStream_t st = prio_stream(0);
  printf("{\"mode\": \"deferred\", \"layers_timed\": %d, ", PASS_LAYERS);
  print_pass(p);
  printf(", \"rejected\": %d, \"missing_algo\": %d, \"arms\": {", g_rejected_total, g_missing);
  const double macs = (p.lin_macs + p.att_macs) * PASS_LAYERS;
  int i = 0;
  for (bool forming : {false, true}) {
    const Graph G = capture(st, [&] { for (int l = 0; l < PASS_LAYERS; l++) pass_layer(d, p, l, forming, st); });
    print_reps(forming ? "form" : "gemm", time_graph(d, G, st, reps), macs, d.sms, i++ == 0);
    CK(cudaGraphExecDestroy(G.ge));
  }
  // the linears alone and attention alone
  {
    Pass q = p;
    q.seqs.clear();
    const Graph G = capture(st, [&] { for (int l = 0; l < PASS_LAYERS; l++) pass_layer(d, q, l, false, st); });
    printf(", ");
    print_reps("lin", time_graph(d, G, st, reps), p.lin_macs * PASS_LAYERS, d.sms, true);
    CK(cudaGraphExecDestroy(G.ge));
    std::vector<Rep> att_lt, att_fu;
    for (bool fused : {false, true}) {
      const Graph A = capture(st, [&] {
        for (int l = 0; l < PASS_LAYERS; l++)
          for (const Seq& s : p.seqs) {
            const int m = 4 * s.t;
            if (fused) {
              att_run(p.Q, p.Qn, s.K, s.V, m, s.lk, p.O, p.csum, 0x304, st);
              continue;
            }
            const LtB& QK = plan(m, s.lk, HD, HKV, 0, 0, 0, nullptr, nullptr, nullptr, d.st);
            const LtB& PV = plan(m, HD, s.lk, HKV, 0, 0, 0, nullptr, nullptr, nullptr, d.st);
            ltb_run(QK, p.lt, p.Q, s.K, p.S, st);
            ltb_run(QK, p.lt, p.Q, s.K, p.S, st);
            ltb_run(PV, p.lt, p.P, s.V, p.O, st);
          }
      });
      (fused ? att_fu : att_lt) = time_graph(d, A, st, reps);
      CK(cudaGraphExecDestroy(A.ge));
    }
    print_reps("att", g_att_fused ? att_fu : att_lt, p.att_macs * PASS_LAYERS, d.sms, true);
    print_reps("att_lt", att_lt, p.att_macs * PASS_LAYERS, d.sms, true);
    print_reps("att_fused", att_fu, p.att_macs * PASS_LAYERS, d.sms, false);
  }
  printf("}}\n");
  fflush(stdout);
  return 0;
}

// a host thread keeps `depth` launches of graph G queued on stream fs and notes when each is seen done
struct PumpG {
  std::thread th;
  std::atomic<bool> on{false};
  std::atomic<long long> done{0};
  std::vector<double> t;
};
static void pumpg_start(PumpG& p, const Graph& G, cudaStream_t fs, int depth) {
  p.on = true;
  p.done = 0;
  p.t.clear();
  p.th = std::thread([&p, &G, fs, depth] {
    std::vector<cudaEvent_t> ev(depth);
    for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync));
    for (long long k = 0; p.on.load(); k++) {
      const int slot = (int)(k % depth);
      if (k >= depth) {
        CK(cudaEventSynchronize(ev[slot]));
        p.t.push_back(now_ms());
        p.done++;
      }
      CK(cudaGraphLaunch(G.ge, fs));
      CK(cudaEventRecord(ev[slot], fs));
    }
    CK(cudaStreamSynchronize(fs));
    for (auto& e : ev) CK(cudaEventDestroy(e));
  });
}
static void pumpg_stop(PumpG& p) {
  p.on = false;
  p.th.join();
}
static long long items_in(const PumpG& p, double t0, double t1) {
  long long n = 0;
  for (double x : p.t) n += x > t0 && x <= t1;
  return n;
}
// the step on sst, and beside each step per_step of the pass's layers (credit carried across steps), so the pass gets exactly
// the token-layers decode produces; the mean step over whole periods, and how long the pass's stream takes to drain after
// the last step (a pass that can't keep up leaves a backlog that grows with the reps)
static std::vector<Rep> time_paced(Dev& d, const std::function<void()>& step, cudaStream_t sst, const Graph& Gp, cudaStream_t pst,
                                   double per_step, int reps, long long* launched, double* tail_ms) {
  d.st = sst;
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(sst, cudaStreamCaptureModeThreadLocal));
  step();
  CK(cudaStreamEndCapture(sst, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 2; i++) CK(cudaGraphLaunch(ge, sst));
  CK(cudaStreamSynchronize(sst));
  cudaEvent_t e0, e1, ep;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventCreate(&ep));
  std::vector<Rep> out;
  double credit = 0;
  long long n = 0;
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(e0, sst));
    CK(cudaGraphLaunch(ge, sst));
    CK(cudaEventRecord(e1, sst));
    credit += per_step;
    while (credit >= 1 - 1e-9) {
      CK(cudaGraphLaunch(Gp.ge, pst));
      credit -= 1;
      n++;
    }
    double cs = 0, ps = 0;
    int nc = 0, np = 0;
    while (cudaEventQuery(e1) == cudaErrorNotReady) {
      const double c = nvml_mhz(d), p = nvml_w(d);
      if (c > 0) { cs += c; nc++; }
      if (p > 0) { ps += p; np++; }
    }
    CK(cudaEventSynchronize(e1));
    if (nc == 0) { cs = nvml_mhz(d); nc = 1; }
    if (np == 0) { ps = nvml_w(d); np = 1; }
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    out.push_back({ms, cs / nc, ps / np});
  }
  const double t = now_ms();
  CK(cudaEventRecord(ep, pst));
  CK(cudaEventSynchronize(ep));
  *tail_ms = now_ms() - t;
  *launched = n;
  CK(cudaGetLastError());
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  CK(cudaEventDestroy(ep));
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  return out;
}
static int run_beside(Dev& d, int T, bool b32, int B, int L, int fill_sms, bool forming, int reps) {
  Pass p = make_pass(d, T, b32);
  pass_layer(d, p, 0, forming, d.st, true);
  const std::vector<Lt> lp = lin_plans(d, B);
  bool wrapped = false;
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t full = d.st, s_hi = prio_stream(hi), s_lo = prio_stream(lo);
  auto step = [&] { enqueue_step(d, lp, 2, B, L, &wrapped); };
  auto pass_graph = [&](cudaStream_t st) {
    return capture(st, [&] { pass_layer(d, p, 1, forming, st); });
  };
  const double pass_macs = p.lin_macs + p.att_macs;
  // the pass alone on the whole GPU
  const Graph Gp = pass_graph(s_lo);
  const std::vector<Rep> pa = time_graph(d, Gp, s_lo, std::max(3, reps / 3));
  d.st = full;
  const std::vector<Rep> a_full = time_work(d, step, reps);
  // the step at high priority, the pass kept queued at low priority
  d.st = s_hi;
  PumpG pump;
  pumpg_start(pump, Gp, s_lo, 2);
  while (pump.done.load() < 1) {
  }
  double t0 = 0, t1 = 0;
  const std::vector<Rep> a_prio = time_work(d, step, reps, &t0, &t1);
  pumpg_stop(pump);
  const long long n_prio = items_in(pump, t0, t1);
  const double w_prio = t1 - t0;
  const double per_step = (double)B * LAYERS / T;
  const int preps = std::min(256, std::max(reps, (int)std::ceil(4.0 / per_step)));
  long long pl_prio = 0, pl_green = 0;
  double tail_prio = 0, tail_green = 0;
  const std::vector<Rep> a_pprio = time_paced(d, step, s_hi, Gp, s_lo, per_step, preps, &pl_prio, &tail_prio);
  // a green split: the step on the rest, the pass on fill_sms SMs
  cudaStream_t ss = nullptr, fs = nullptr;
  int got[2] = {0, 0};
  const int e = gc_split(d.sms - fill_sms, &ss, &fs, got);
  std::vector<Rep> a_split, a_green, a_pgreen;
  long long n_green = 0;
  double w_green = 0;
  std::vector<Rep> pg;
  if (!e && fs) {
    d.st = ss;
    a_split = time_work(d, step, reps);
    const Graph Gg = pass_graph(fs);
    pg = time_graph(d, Gg, fs, std::max(3, reps / 3));
    pumpg_start(pump, Gg, fs, 2);
    while (pump.done.load() < 1) {
    }
    a_green = time_work(d, step, reps, &t0, &t1);
    pumpg_stop(pump);
    n_green = items_in(pump, t0, t1);
    w_green = t1 - t0;
    a_pgreen = time_paced(d, step, ss, Gg, fs, per_step, preps, &pl_green, &tail_green);
    CK(cudaGraphExecDestroy(Gg.ge));
  } else {
    fprintf(stderr, "gc_split(%d): %d\n", d.sms - fill_sms, e);
  }
  d.st = full;
  const std::vector<Rep> a_full2 = time_work(d, step, reps);
  double step_macs = 0;
  for (int j = 0; j < 4; j++) step_macs += (double)LAYERS * B * LN[j] * LK[j];
  step_macs += (double)B * VOCAB * HID + 2.0 * LAYERS * B * HQ * HD * (double)L;
  const double base = (median(a_full).ms + median(a_full2).ms) / 2;
  printf("{\"mode\": \"beside\", \"B\": %d, \"L\": %d, \"forming\": %s, \"fill_sms_req\": %d, \"step_sms\": %d, \"fill_sms\": %d, "
         "\"kv_wrapped\": %s, \"step_macs\": %.6e, \"pass_macs_layer\": %.6e, ",
         B, L, forming ? "true" : "false", fill_sms, got[0], got[1], wrapped ? "true" : "false", step_macs, pass_macs);
  print_pass(p);
  printf(", \"rejected\": %d, \"missing_algo\": %d", g_rejected_total, g_missing);
  printf(", \"prio\": {\"window_ms\": %.3f, \"pass_layers\": %lld, \"slowdown\": %.4f}", w_prio, n_prio, median(a_prio).ms / base);
  printf(", \"green\": {\"window_ms\": %.3f, \"pass_layers\": %lld, \"slowdown\": %.4f, \"slowdown_vs_split\": %.4f}", w_green, n_green,
         a_green.empty() ? 0.0 : median(a_green).ms / base,
         a_green.empty() ? 0.0 : median(a_green).ms / median(a_split).ms);
  const double base_mean = (mean_ms(a_full) + mean_ms(a_full2)) / 2;
  printf(", \"paced\": {\"per_step\": %.6f, \"reps\": %d, \"prio\": {\"pass_layers\": %lld, \"tail_ms\": %.3f, \"slowdown_mean\": %.4f}, "
         "\"green\": {\"pass_layers\": %lld, \"tail_ms\": %.3f, \"slowdown_mean\": %.4f}}",
         per_step, preps, pl_prio, tail_prio, mean_ms(a_pprio) / base_mean, pl_green, tail_green,
         a_pgreen.empty() ? 0.0 : mean_ms(a_pgreen) / base_mean);
  printf(", \"arms\": {");
  print_reps("pass_alone", pa, pass_macs, d.sms, true);
  if (!pg.empty()) print_reps("pass_alone_green", pg, pass_macs, d.sms, true);
  print_reps("full", a_full, step_macs, d.sms, true);
  print_reps("prio", a_prio, step_macs, d.sms, true);
  print_reps("paced_prio", a_pprio, step_macs, d.sms, true);
  if (!a_split.empty()) print_reps("split", a_split, step_macs, d.sms, true);
  if (!a_green.empty()) print_reps("green", a_green, step_macs, d.sms, true);
  if (!a_pgreen.empty()) print_reps("paced_green", a_pgreen, step_macs, d.sms, true);
  print_reps("full2", a_full2, step_macs, d.sms, false);
  printf("}}\n");
  fflush(stdout);
  CK(cudaGraphExecDestroy(Gp.ge));
  return 0;
}

// ---------------------------------------------------------------- the gate: commitments, then dcap's GEMM and attention gate
// att_fused against the host on every head and row (a partial CTA among them): O = (att_q(Q K^T) + noise) V and the
// checksum a row of Qn K^T
static int att_gate(Dev& d) {
  const int rows = 160, lk = 96;
  const uint32_t seed = 0x1234;
  const size_t nq = (size_t)HKV * rows * HD, nkv = (size_t)HKV * lk * HD;
  int8_t *Q, *Qn, *K, *V;
  int *O, *cs;
  CK(cudaMalloc(&Q, nq)); CK(cudaMalloc(&Qn, nq)); CK(cudaMalloc(&K, nkv)); CK(cudaMalloc(&V, nkv));
  CK(cudaMalloc(&O, nq * sizeof(int))); CK(cudaMalloc(&cs, (size_t)HKV * rows * sizeof(int)));
  fill_s8<<<64, 256, 0, d.st>>>(Q, nq, 0xA1); fill_s8<<<64, 256, 0, d.st>>>(Qn, nq, 0xA2);
  fill_s8<<<64, 256, 0, d.st>>>(K, nkv, 0xA3); fill_s8<<<64, 256, 0, d.st>>>(V, nkv, 0xA4);
  std::vector<int8_t> q(nq), qn(nq), k(nkv), v(nkv);
  CK(cudaStreamSynchronize(d.st));
  CK(cudaMemcpy(q.data(), Q, nq, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(qn.data(), Qn, nq, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(k.data(), K, nkv, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(v.data(), V, nkv, cudaMemcpyDeviceToHost));
  int fails = 0;
  for (int ns : {1, 2, 3}) {
    CK(cudaMemsetAsync(O, 0xA5, nq * sizeof(int), d.st)); CK(cudaMemsetAsync(cs, 0xA5, (size_t)HKV * rows * sizeof(int), d.st));
    att_run(Q, Qn, K, V, rows, lk, O, cs, seed, d.st, ns);
    CK(cudaStreamSynchronize(d.st));
    CK(cudaGetLastError());
    std::vector<int> o(nq), c((size_t)HKV * rows);
    CK(cudaMemcpy(o.data(), O, o.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.data(), cs, c.size() * sizeof(int), cudaMemcpyDeviceToHost));
    long bad_o = 0, bad_c = 0;
    std::vector<int8_t> p(lk);
    for (int h = 0; h < HKV; h++)
      for (int r = 0; r < rows; r++) {
        const int8_t* qr = &q[((size_t)h * rows + r) * HD];
        const int8_t* qnr = &qn[((size_t)h * rows + r) * HD];
        uint32_t sum = 0;
        for (int key = 0; key < lk; key++) {
          const int8_t* kr = &k[((size_t)h * lk + key) * HD];
          int s = 0, sn = 0;
          for (int e = 0; e < HD; e++) { s += qr[e] * kr[e]; sn += qnr[e] * kr[e]; }
          sum += (uint32_t)sn;
          const uint32_t u = att_noise(seed, (uint32_t)r, (uint32_t)(key & ~3));
          p[key] = (int8_t)(uint8_t)(att_q(s) + ((u >> (8 * (key & 3))) & 0xff));
        }
        bad_c += (int)sum != c[(size_t)h * rows + r];
        for (int e = 0; e < HD; e++) {
          int acc = 0;
          for (int key = 0; key < lk; key++) acc += p[key] * v[((size_t)h * HD + e) * lk + key];
          bad_o += acc != o[((size_t)h * rows + r) * HD + e];
        }
      }
    printf("{\"mode\": \"gate\", \"what\": \"att_fused\", \"rows\": %d, \"lk\": %d, \"splits\": %d, \"o_mismatches\": %ld, "
           "\"csum_mismatches\": %ld, \"ok\": %s}\n", rows, lk, ns, bad_o, bad_c, bad_o || bad_c ? "false" : "true");
    fails += bad_o || bad_c;
  }
  CK(cudaFree(Q)); CK(cudaFree(Qn)); CK(cudaFree(K)); CK(cudaFree(V)); CK(cudaFree(O)); CK(cudaFree(cs));
  return fails;
}
static int run_gate(Dev& d, const std::string& dir, const std::vector<int>& Ms) {
  int fails = att_gate(d);
  for (int M : Ms)
    for (int t = 0; t < NSTAGE; t++) {
      if (t == 4 || t == 3) continue;  // the same width as stage 0
      const HostConst c = load_const(dir, SK[t], M);
      const Commit m = make_commit(c);
      std::vector<uint8_t> t_sz, t_fr, t_mid, t_slab;
      const bool ok = commit_gate(m, 1, &t_sz);
      g_mid = 0;
      const bool fast = commit_gate(m, 2, &t_fr);
      g_mid = 1;
      const bool mid = commit_gate(m, 2, &t_mid), slab = commit_gate(m, 2, &t_slab, 4), control = commit_gate(m, 0);
      std::vector<uint8_t> t_pair;
      g_pair = 1;
      const bool pair = commit_gate(m, 2, &t_pair, 4), pair_host = commit_gate(m, 2, nullptr, 4, 16);
      g_pair = 0;
      std::vector<uint8_t> t_c8, t_c3;
      const bool col8 = commit_gate(m, 2, &t_c8, 1, 0, 8), col3 = commit_gate(m, 2, &t_c3, 1, 0, 3);
      pool_start(16);
      const bool col8_host = commit_gate(m, 2, nullptr, 1, 512, 8);
      pool_stop();
      const bool same = t_sz == t_fr && t_sz == t_mid && t_sz == t_slab && t_sz == t_pair && t_sz == t_c8 && t_sz == t_c3;
      printf("{\"mode\": \"gate\", \"what\": \"hm96-sha512 row-seg frame-v3-sha512\", \"k\": %d, \"R\": %d, \"s\": %d, \"ok\": %s, "
             "\"fast_ok\": %s, \"mid_ok\": %s, \"slab4_ok\": %s, \"pair_slab4_ok\": %s, \"pair_slab4_E16_ok\": %s, "
             "\"cols8_ok\": %s, \"cols3_ok\": %s, \"cols8_E512_P16_ok\": %s, "
             "\"fast_tree_equal\": %s, \"control_rejected\": %s, \"host_top\": {",
             c.k, c.R, c.s, ok ? "true" : "false", fast ? "true" : "false", mid ? "true" : "false", slab ? "true" : "false",
             pair ? "true" : "false", pair_host ? "true" : "false", col8 ? "true" : "false", col3 ? "true" : "false",
             col8_host ? "true" : "false", same ? "true" : "false", control ? "false" : "true");
      fails += !ok || !fast || !mid || !slab || !pair || !pair_host || !col8 || !col3 || !col8_host || !same || control;
      bool fe = true;
      for (int E : {1, 16, 32, 64, 128})
        for (int k : {1, 4}) {
          const bool h = commit_gate(m, 2, nullptr, k, E);
          printf("%s\"E%d_k%d\": %s", fe ? "" : ", ", E, k, h ? "true" : "false");
          fe = false;
          fails += !h;
        }
      for (const auto& ep : {std::make_pair(256, 8), std::make_pair(512, 16), std::make_pair(1024, 32)}) {
        pool_start(ep.second);
        for (int k : {1, 4}) {
          const bool h = commit_gate(m, 2, nullptr, k, ep.first);
          printf(", \"E%d_P%d_k%d\": %s", ep.first, ep.second, k, h ? "true" : "false");
          fails += !h;
        }
        pool_stop();
      }
      printf("}}\n");
    }
  fflush(stdout);
  return fails + gate(d);
}

// the host pool's roots (h_warm, h_top) against one thread's (h_sub) over random templates and exports, on the host alone:
// trees of 64, 512 and 4096 leaves, E up to 1024 (at E >= width the tree's export is its leaves)
static int pool_test(int maxP) {
  for (const char* lib : {"libcrypto.so.3", "libcrypto.so.1.1", "libcrypto.so"})
    if (void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL)) {
      g_ossl = (void (*)(void*, const unsigned char*))dlsym(h, "SHA512_Transform");
      if (g_ossl) break;
    }
  if (!g_ossl) { fprintf(stderr, "pooltest: no libcrypto SHA512_Transform\n"); return 2; }
  g_use_ossl = true;
  uint64_t z = 7;
  auto rnd = [&] {
    z += 0x9E3779B97F4A7C15ull;
    uint64_t x = z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
  };
  std::vector<uint8_t> ex(64 * 1024);
  int cases = 0, bad = 0;
  for (int width : {64, 512, 4096}) {
    HostConst c{};
    c.width = width;
    c.ntpl.resize((size_t)384 * (c.width - 1));
    c.noff.resize((size_t)4 * (c.width - 1));
    for (auto& b : c.ntpl) b = (uint8_t)rnd();
    for (long long j = 0; j < c.width - 1; j++) {
      const uint32_t o = 139 + (uint32_t)(j & 1);
      memcpy(&c.noff[4 * j], &o, 4);
    }
    std::vector<uint64_t> nmid((size_t)8 * (c.width - 1));
    for (auto& x : nmid) x = rnd();
    for (int P = 2; P <= maxP; P *= 2) {
      pool_start(P);
      for (int E : {2 * P, 256, 512, 1024}) {
        if (E < 2 * P) continue;
        const int ew = export_width(width, E);
        for (int it = 0; it < 100; it++) {
          for (int b = 0; b < 64 * ew; b++) ex[b] = (uint8_t)rnd();
          uint8_t want[64], got[64];
          h_sub(c, nmid, ex.data(), ew, ew, 0, want);
          if (it & 1) h_warm(c, nmid, E);
          h_top(c, nmid, ex.data(), E, got);
          bad += memcmp(want, got, 64) != 0;
          cases++;
        }
      }
      pool_stop();
    }
  }
  printf("{\"mode\": \"pooltest\", \"cases\": %d, \"bad\": %d}\n", cases, bad);
  return bad != 0;
}
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: stage gate DIR | prefill DIR M | deferred T seq500|b32 | beside T seq500|b32 B L FILL_SMS form|gemm  [reps] | pooltest [maxP]\n");
    return 1;
  }
  const std::string mode = argv[1];
  if (mode == "pooltest") return pool_test(argc > 2 ? atoi(argv[2]) : 32);
  g_fast = !(getenv("HASH") && strcmp(getenv("HASH"), "sz") == 0);
  g_att_fused = !(getenv("ATT") && strcmp(getenv("ATT"), "lt") == 0);
  Dev d;
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&d.st));
  CK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, 0));
  if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); return 2; }
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
  if (nvmlDeviceGetHandleByPciBusId_v2(bus, &d.nv) != NVML_SUCCESS) { fprintf(stderr, "NVML handle failed\n"); return 2; }
  char uuid[96] = {0}, name[96] = {0};
  nvmlDeviceGetUUID(d.nv, uuid, sizeof uuid);
  nvmlDeviceGetName(d.nv, name, sizeof name);
  const char* want = getenv("GPU_LEASE_UUID");
  if (want && *want && strncmp(want, uuid, strlen(uuid)) != 0) {
    fprintf(stderr, "device %s is not the leased %s\n", uuid, want);
    return 3;
  }
  fprintf(stderr, "device %s %s, %d SMs\n", name, uuid, d.sms);
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  CK(cudaMalloc(&g_ref, (size_t)256 * VOCAB * sizeof(int)));
  CK(cudaMalloc(&g_cnt, sizeof *g_cnt));
  for (int l = 0; l < LAYERS; l++)
    for (int j = 0; j < 4; j++) {
      const size_t n = (size_t)LN[j] * LK[j];
      CK(cudaMalloc(&d.W[l][j], n));
      fill_s8<<<1024, 256, 0, d.st>>>(d.W[l][j], n, 0x1000 + l * 4 + j);
    }
  CK(cudaMalloc(&d.Wlm, (size_t)VOCAB * HID));
  fill_s8<<<1024, 256, 0, d.st>>>(d.Wlm, (size_t)VOCAB * HID, 0x77);
  CK(cudaMalloc(&d.X, (size_t)8192 * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(d.X, (size_t)8192 * FF, 0x55);
  CK(cudaMalloc(&d.D, (size_t)1 << 30));
  const size_t maxBL = 786432;
  d.kv_bytes = (size_t)1 << 30;
  if (mode == "beside" && argc > 6) d.kv_bytes = (size_t)atoi(argv[4]) * atoi(argv[5]) * KV_TOKEN_BYTES + ((size_t)1 << 20);
  CK(cudaMalloc(&d.kv, d.kv_bytes));
  fill_s8<<<4096, 256, 0, d.st>>>(d.kv, d.kv_bytes, 0x99);
  CK(cudaMalloc(&d.q, (size_t)256 * HQ * HD));
  fill_s8<<<64, 256, 0, d.st>>>(d.q, (size_t)256 * HQ * HD, 0x31);
  CK(cudaMalloc(&d.P, maxBL * HQ));
  fill_s8<<<1024, 256, 0, d.st>>>(d.P, maxBL * HQ, 0x32);
  CK(cudaMalloc(&d.S, maxBL * HQ * sizeof(int)));
  CK(cudaMalloc(&d.O, (size_t)256 * HQ * HD * sizeof(int)));
  CK(cudaMalloc(&d.sink, 64));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  h_init();
  link_init();
  g_mid = !(getenv("MID") && atoi(getenv("MID")) == 0);
  int rc = 0;
  if (mode == "shalat") {
    sha_latency(d);
  } else if (mode == "gate") {
    if (argc < 3) return 1;
    hm_load(argv[2]);
    std::vector<int> Ms;
    for (int i = 3; i < argc; i++) Ms.push_back(atoi(argv[i]));
    if (Ms.empty()) Ms = {2048, 8192};
    rc = run_gate(d, argv[2], Ms);
  } else if (mode == "prefill") {
    if (argc < 4) return 1;
    hm_load(argv[2]);
    rc = run_prefill(d, argv[2], atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 5);
  } else if (mode == "deferred") {
    if (argc < 4) return 1;
    rc = run_deferred(d, atoi(argv[2]), strcmp(argv[3], "b32") == 0, argc > 4 ? atoi(argv[4]) : 5);
  } else if (mode == "beside") {
    if (argc < 8) return 1;
    const int B = atoi(argv[4]), L = atoi(argv[5]);
    if ((size_t)B * L > maxBL || L % 512 || B > 256) { fprintf(stderr, "B*L <= %zu, L %% 512 == 0, B <= 256\n", maxBL); return 1; }
    rc = run_beside(d, atoi(argv[2]), strcmp(argv[3], "b32") == 0, B, L, atoi(argv[6]), strcmp(argv[7], "form") == 0,
                    argc > 8 ? atoi(argv[8]) : 21);
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }
  return rc;
}
