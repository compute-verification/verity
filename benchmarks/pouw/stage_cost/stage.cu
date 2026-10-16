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
#include "../decode_capacity/dcap.cu"
#undef main
#include "../served_zk/hm96_rows.cu"

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
  if (c.s < 1 || c.width > 128 * 128) { fprintf(stderr, "fr_tree takes s >= 1 and at most 128 CTAs of 128 leaves\n"); exit(2); }
  const FrConst& f = g_fr[key];
  m.ltpl = f.ltpl; m.loff = f.loff; m.ntpl = f.ntpl; m.noff = f.noff;
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
// s512_compress with 16 rounds unrolled, not 80: the latency path runs a warp or few a scheduler, where fully unrolled code
// (~128 KB a compression) is bound by instruction fetch
__device__ __forceinline__ void fr_compress(uint64_t st[8], uint64_t w[16]) {
  uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll 1
  for (int r = 0; r < 80; r += 16) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
      if (r) {
        const uint64_t w15 = w[(j + 1) & 15], w2 = w[(j + 14) & 15];
        w[j] += (s512_rotr(w2, 19) ^ s512_rotr(w2, 61) ^ (w2 >> 6)) + w[(j + 9) & 15]
              + (s512_rotr(w15, 1) ^ s512_rotr(w15, 8) ^ (w15 >> 7));
      }
      const uint64_t t1 = h + (s512_rotr(e, 14) ^ s512_rotr(e, 18) ^ s512_rotr(e, 41)) + ((e & f) ^ (~e & g)) + SHA512_K[r + j] + w[j];
      const uint64_t t2 = (s512_rotr(a, 28) ^ s512_rotr(a, 34) ^ s512_rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
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
// sz_segments with fr_compress (the same digests)
extern "C" __global__ void fr_segments(Args a) {
  extern __shared__ uint32_t fr_sw[];
  const uint8_t* data = (const uint8_t*)a.p[0];
  const uint64_t* mids = (const uint64_t*)a.p[1];
  uint8_t* out = (uint8_t*)a.p[2];
  long long rows = a.i[0], stride = a.i[1];
  int s = a.i[2], last = a.i[3];
  long long total = rows * s;
  int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t* sh = fr_sw + wid * (32 * 33);
  long long warp0 = ((long long)blockIdx.x * (blockDim.x >> 5) + wid) * 32;
  if (warp0 >= total) return;
  long long g = warp0 + lane;
  int j = (int)(g % s);
  int bytes = j == s - 1 ? last : 1024;
  uint64_t st[8];
  if (g < total)
    for (int q = 0; q < 8; q++) st[q] = mids[8 * j + q];
#pragma unroll 1
  for (int c = 0; c < 8; c++) {
#pragma unroll
    for (int it = 0; it < 8; it++) {
      int idx = it * 32 + lane, piece = idx >> 3, off = idx & 7;
      long long pg = warp0 + piece;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (pg < total) {
        int pj = (int)(pg % s);
        int pb = pj == s - 1 ? last : 1024;
        if (c * 128 < pb) v = *(const uint4*)(data + (pg / s) * stride + (long long)pj * 1024 + c * 128 + off * 16);
      }
      uint32_t* d = sh + piece * 33 + off * 4;
      d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    }
    __syncwarp();
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
// SHA-512 of a padded template of nblk blocks, value L at byte o (and R at o + 72, a node's right child)
template <bool NODE>
__device__ __forceinline__ void fr_hash(const uint8_t* tpl, int nblk, int o, const uint64_t L[8], const uint64_t R[8], uint64_t st[8]) {
  s512_iv(st);
#pragma unroll 1
  for (int b = 0; b < nblk; b++) {
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
// level above, the current level being lw nodes wide at tree offset off with its parents' templates from nb
__device__ __noinline__ void fr_level(uint64_t (*sh)[8], int n, long long j0, long long lw, long long off, long long nb,
                                      const uint8_t* ntpl, const uint32_t* noff, uint8_t* tree) {
  const int t = threadIdx.x;
  const bool act = t < n / 2;
  uint64_t v[8];
  if (act) {
    uint64_t L[8], R[8];
#pragma unroll
    for (int k = 0; k < 8; k++) { L[k] = sh[2 * t][k]; R[k] = sh[2 * t + 1][k]; }
    const long long j = j0 / 2 + t, idx = nb + j;
    fr_hash<true>(ntpl + idx * 384, 3, (int)noff[idx], L, R, v);
    s512_out(v, tree + (off + lw + j) * 64);
  }
  __syncthreads();
  if (act)
#pragma unroll
    for (int k = 0; k < 8; k++) sh[t][k] = v[k];
  __syncthreads();
}
// p: digests, q_mid, my, cs (the layer's slot), hm96 mids, leaf tpl, leaf off, node tpl, node off, hm96 leaves out, b || c
//    (b written), tree, counter   i: rows, s, width.  Grid width / L CTAs of L = min(width, 128) leaves, at most 128 CTAs.
extern "C" __global__ void __launch_bounds__(128) fr_tree(Args a) {
  __shared__ uint64_t sh[128][8];
  __shared__ int last;
  const int tid = threadIdx.x, L = blockDim.x, s = a.i[1];
  const long long R = a.i[0], width = a.i[2], i = (long long)blockIdx.x * L + tid;
  const uint8_t* ntpl = (const uint8_t*)a.p[7];
  const uint32_t* noff = (const uint32_t*)a.p[8];
  uint8_t* tree = (uint8_t*)a.p[11];
  uint64_t v[8];
  if (i < R) {
    const uint64_t* q = (const uint64_t*)a.p[1];
    uint64_t st[8], w[16];
#pragma unroll
    for (int t = 0; t < 8; t++) st[t] = q[t];
    const uint4* dg = (const uint4*)((const uint8_t*)a.p[0] + i * (long long)s * 64);
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
    const uint64_t* my = (const uint64_t*)a.p[2] + 8 * i;
    const uint64_t* cs = (const uint64_t*)a.p[3] + 8 * i;
    uint64_t* bo = (uint64_t*)((uint8_t*)a.p[10] + i * 128);
    const uint64_t* mid = (const uint64_t*)a.p[4];
    uint64_t lt[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const uint64_t b = s512_bswap(st[k]) ^ my[k];
      bo[k] = b;
      w[k] = s512_bswap(b);
      w[8 + k] = cs[k];
      lt[k] = mid[8 + k];
    }
    fr_compress(lt, w);
    fr_pad_block(lt, 256);
    s512_out(lt, (uint8_t*)a.p[9] + i * 64);
    fr_hash<false>((const uint8_t*)a.p[5] + i * 256, 2, (int)((const uint32_t*)a.p[6])[i], lt, lt, v);
    s512_out(v, tree + i * 64);
  } else {
    const uint64_t* pd = (const uint64_t*)(tree + i * 64);
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = s512_bswap(pd[k]);
  }
#pragma unroll
  for (int k = 0; k < 8; k++) sh[tid][k] = v[k];
  __syncthreads();
  long long lw = width, off = 0, nb = 0, j0 = (long long)blockIdx.x * L;
  for (int n = L; n > 1; n >>= 1) {
    fr_level(sh, n, j0, lw, off, nb, ntpl, noff, tree);
    nb += lw / 2; off += lw; lw /= 2; j0 /= 2;
  }
  if (gridDim.x == 1) return;
  int* counter = (int*)a.p[12];
  if (tid == 0) {
    __threadfence();
    last = atomicAdd(counter, 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int n0 = (int)gridDim.x;  // == lw: the CTAs' subtree roots
  if (tid < n0) {
    const uint64_t* pd = (const uint64_t*)(tree + (off + tid) * 64);
#pragma unroll
    for (int k = 0; k < 8; k++) sh[tid][k] = s512_bswap(__ldcg(pd + k));
  }
  __syncthreads();
  j0 = 0;
  for (int n = n0; n > 1; n >>= 1) {
    fr_level(sh, n, j0, lw, off, nb, ntpl, noff, tree);
    nb += lw / 2; off += lw; lw /= 2;
  }
  if (tid == 0) *counter = 0;
}
static bool g_fast = true;  // HASH=sz: served_zk's hm96_rows.cu as is
// the salt-only parts of `slots` layers' commitments from slot 0, in one launch
static void commit_salt(const Commit& m, int slots, cudaStream_t st) {
  Args a{};
  const unsigned long long ps[6] = {(unsigned long long)g_tables, (unsigned long long)m.salts, (unsigned long long)g_hm_mid,
                                    (unsigned long long)m.my, (unsigned long long)m.cs, (unsigned long long)m.bc};
  for (int j = 0; j < 6; j++) a.p[j] = ps[j];
  const long long rows = (long long)m.R * slots;
  a.i[0] = (int)rows; a.i[1] = m.R;
  fr_salt<<<(unsigned)((rows + 127) / 128), 128, 0, st>>>(a);
}
static void commit_segs(const Commit& m, const uint8_t* data, cudaStream_t st) {
  Args a{};
  a.p[0] = (unsigned long long)data;
  a.p[1] = (unsigned long long)m.mids;
  a.p[2] = (unsigned long long)m.digests;
  a.i[0] = m.R; a.i[1] = m.k; a.i[2] = m.s; a.i[3] = m.last;
  const long long total = (long long)m.R * m.s;
  fr_segments<<<(unsigned)((total + 255) / 256), 256, 8 * 32 * 33 * 4, st>>>(a);
}
static void commit_frtree(const Commit& m, int slot, cudaStream_t st) {
  Args f{};
  const unsigned long long ps[13] = {(unsigned long long)m.digests, (unsigned long long)(m.mids + 64 * m.s),
                                     (unsigned long long)(m.my + (size_t)64 * m.R * slot),
                                     (unsigned long long)(m.cs + (size_t)64 * m.R * slot), (unsigned long long)g_hm_mid,
                                     (unsigned long long)m.ltpl, (unsigned long long)m.loff, (unsigned long long)m.ntpl,
                                     (unsigned long long)m.noff, (unsigned long long)m.x, (unsigned long long)m.bc,
                                     (unsigned long long)m.tree, (unsigned long long)m.counter};
  for (int j = 0; j < 13; j++) f.p[j] = ps[j];
  f.i[0] = m.R; f.i[1] = m.s; f.i[2] = m.width;
  const int L = std::min(m.width, 128);
  fr_tree<<<(unsigned)(m.width / L), L, 0, st>>>(f);
}
static void commit_tree(const Commit& m, int slot, const uint8_t* data, cudaStream_t st) {
  commit_segs(m, data, st);
  commit_frtree(m, slot, st);
}
// n chained compressions a thread (each block xor-fed from the last state), rolled (fr_compress) or fully unrolled
__global__ void sha_chain(int n, int unrolled, uint64_t* out) {
  uint64_t st[8], w[16];
  s512_iv(st);
  for (int j = 0; j < 16; j++) w[j] = (uint64_t)(threadIdx.x + 1) * (j + 1);
  for (int i = 0; i < n; i++) {
    if (unrolled) s512_compress(st, w);
    else fr_compress(st, w);
    for (int j = 0; j < 8; j++) w[j] ^= st[j];
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = st[0];
}
// the latency of one compression in a chain: (n = 257) - (n = 1) over 256, by threads a CTA (1: one thread on the GPU)
static void sha_latency(Dev& d) {
  uint64_t* out;
  CK(cudaMalloc(&out, 128 * 188 * sizeof(uint64_t)));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  printf("{\"mode\": \"sha_latency\", \"mhz\": %.0f, \"us_a_compression\": {", nvml_mhz(d));
  bool first = true;
  for (int unrolled : {0, 1})
    for (int thr : {1, 32, 128}) {
      double ms[2];
      for (int k = 0; k < 2; k++) {
        const int n = k ? 257 : 1;
        std::vector<double> v;
        for (int r = 0; r < 5; r++) {
          CK(cudaEventRecord(e0, d.st));
          sha_chain<<<1, thr, 0, d.st>>>(n, unrolled, out);
          CK(cudaEventRecord(e1, d.st));
          CK(cudaEventSynchronize(e1));
          float x;
          CK(cudaEventElapsedTime(&x, e0, e1));
          v.push_back(x);
        }
        std::sort(v.begin(), v.end());
        ms[k] = v[2];
      }
      printf("%s\"%s_%d\": %.3f", first ? "" : ", ", unrolled ? "unrolled" : "rolled", thr, (ms[1] - ms[0]) * 1e3 / 256);
      first = false;
    }
  printf("}}\n");
  fflush(stdout);
  CK(cudaFree(out));
}
// the commitment of the gate's rows into poisoned buffers by path (1 hm96_rows.cu, 2 fast; 0 the no-write control) equals
// Python's leaves, b || c and root; the whole tree is copied out for the paths to be compared node by node
static bool commit_gate(const Commit& m, int path, std::vector<uint8_t>* tree_out = nullptr) {
  const HostConst& c = *m.hc;
  uint8_t* rows = dput(c.rows.data(), c.rows.size());
  CK(cudaMemset(m.x, 0xA5, (size_t)64 * m.R));
  CK(cudaMemset(m.bc, 0xA5, (size_t)128 * m.R));
  CK(cudaMemset(m.tree, 0xA5, m.tree_bytes));
  CK(cudaMemset(m.my, 0xA5, (size_t)64 * m.R * LAYERS));
  CK(cudaMemset(m.cs, 0xA5, (size_t)64 * m.R * LAYERS));
  if (!c.pads.empty()) CK(cudaMemcpy(m.tree + (size_t)64 * c.R, c.pads.data(), c.pads.size(), cudaMemcpyHostToDevice));
  if (path == 1) commit_run(m, rows, 0);
  if (path == 2) {
    commit_salt(m, LAYERS, 0);
    commit_tree(m, LAYERS - 1, rows, 0);
  }
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint8_t> hl((size_t)64 * m.R), bc((size_t)128 * m.R), root(64);
  CK(cudaMemcpy(hl.data(), m.x, hl.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(bc.data(), m.bc, bc.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(root.data(), m.tree + m.tree_bytes - 64, 64, cudaMemcpyDeviceToHost));
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
  add("gemm", sm, [&] { gemms(ch[0], sm); }, macs);
  add("hash", sm, [&] { hashes(ch[0], sm); }, 0);
  if (g_fast) {
    add("salt", sm, [&] { for (int t = 0; t < NSTAGE; t++) salt(ch[0], t, sm); }, 0);
    add("tree", sm, [&] {
      for (int l = 0; l < LAYERS; l++)
        for (int t = 0; t < NSTAGE; t++) commit_tree(ch[0].cm[t], l, ch[0].op[t], sm);
    }, 0);
  }
  add("serial", sm, [&] { stage_seq(ch[0], sm, true, false, nullptr); }, macs);
  add("conc", sm, [&] {
    fork(sm, s0);
    gemms(ch[0], s0);
    fork(sm, s1);
    hashes(ch[1], s1);
    fork(s0, sm);
    fork(s1, sm);
  }, macs);
  cudaStream_t s1lo = prio_stream(lo), s0hi = prio_stream(hi);
  add("conc_lo", sm, [&] {
    fork(sm, s0hi);
    gemms(ch[0], s0hi);
    fork(sm, s1lo);
    hashes(ch[1], s1lo);
    fork(s0hi, sm);
    fork(s1lo, sm);
  }, macs);
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
  // C chunks in flight, each hash -> draw -> GEMM a stage, chunk 0 at the highest priority
  for (int C = 1; C <= NC; C++) {
    add("pipe" + std::to_string(C), sm, [&] {
      for (int c = 0; c < C; c++) {
        fork(sm, cs[c]);
        stage_seq(ch[c], cs[c], true, true, side[c]);
      }
      for (int c = 0; c < C; c++) fork(cs[c], sm);
    }, C * macs);
  }
  add("pipe1_draw", sm, [&] { stage_seq(ch[0], sm, false, true, nullptr); }, macs);
  add("pipe1_hash", sm, [&] { stage_seq(ch[0], sm, true, false, side[0]); }, macs);
  // the commitments on a green partition of N SMs, the GEMMs on the rest (no graph: the two contexts' streams)
  std::vector<std::tuple<int, int, int, double, double, double>> green;  // N asked, gemm SMs, hash SMs, gemm alone, both, hash alone
  for (int N : {28, 44}) {
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
    if (!pg_on) break;
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
  printf("}, \"arms\": {");
  for (size_t i = 0; i < arms.size(); i++) print_reps(arms[i].first.c_str(), arms[i].second, arm_macs[i], d.sms, i + 1 < arms.size());
  printf("}}\n");
  fflush(stdout);
  return 0;
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
      std::vector<uint8_t> t_sz, t_fr;
      const bool ok = commit_gate(m, 1, &t_sz), fast = commit_gate(m, 2, &t_fr), control = commit_gate(m, 0);
      const bool same = t_sz == t_fr;
      printf("{\"mode\": \"gate\", \"what\": \"hm96-sha512 row-seg frame-v3-sha512\", \"k\": %d, \"R\": %d, \"s\": %d, \"ok\": %s, "
             "\"fast_ok\": %s, \"fast_tree_equal\": %s, \"control_rejected\": %s}\n",
             c.k, c.R, c.s, ok ? "true" : "false", fast ? "true" : "false", same ? "true" : "false", control ? "false" : "true");
      fails += !ok || !fast || !same || control;
    }
  fflush(stdout);
  return fails + gate(d);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: stage gate DIR | prefill DIR M | deferred T seq500|b32 | beside T seq500|b32 B L FILL_SMS form|gemm  [reps]\n");
    return 1;
  }
  const std::string mode = argv[1];
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
