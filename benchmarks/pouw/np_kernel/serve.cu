// The lead's kernel (ssn.cu under BLC=2, ROT=1, FOLD=1: blc4a_t64_e7_s1_rot_fold) as a library a serving process calls
// on its own device tensors; `serve_lead.py` binds it as the lead's `LeadBackend`.
//   load      a linear's B' from its int8 weight [N, K] (N whole 256-column tiles), beta_j and its column words
//             (layout.h's FoldCol: s_j from its weight scale and the code C[0..3, j]), formed on the device once
//   ready     the call's buffers, sized from the loaded linears, and the hash streams
//   a_root    a stand-in for root(A_j): hm96-sha512 rows (native_rows.cu, the served committer's leaves under ssn.cu's
//             stand-in constants) of 1 KiB over A_j's m K bytes and their frame tree, its root handed back (the call's
//             stream synchronized: the certifier's coins follow it)
//   call      A' formed on the device from the call's s8 rows (a_k, tile_a's layout), the certifier's feed copied in
//             (row words, G words and H zero words; lane words and column sources beta_j - sigma at stride N), the
//             forming, which also writes each row's fold words (layout.h's FoldRow), and G's copy (form_g), then the GEMM
//             chunk by chunk (ssn.cu's chunk: at most BUDGET bytes of words), the fold serving verity_int8's y bf16
//             [64 ceil(m / 64), N]. Mode 2 exports each chunk's words and hashes them as 64 KiB hm96 rows with their
//             frame tree on a hash stream of their own buffer, beside the GEMM stream; mode 1 exports and hashes nothing;
//             mode 0 keeps the words in registers (the GEMM serving y alone)
//   words_roots  the hash streams drained, the chunks' tree roots since the last call to it
// Measurement only: the committer's keyed root(A_j) and words root are what these stand in for.
//   check_operands  a_k and b_k against ssn.cu's host tile_a and tile_b
// Build: served_lead.sh (with native_rows.cu and its plain-leaf copy, as run.sh builds ssn).
#define main ssn_main
#include "ssn.cu"
#undef main

namespace lead {

constexpr int BN = 256, A_ROW = 1024, W_ROW = 65536, MAX_CHUNKS = 8192;

__device__ __forceinline__ int kidx_d(int t, int l, int K) {
  const int k = t == 0 ? (l < 21 ? l : -1) : (l < 28 ? 21 + 28 * (t - 1) + l : -1);
  return k >= 0 && k < K ? k : -1;
}
// the tile-local position of u32 q of a K-major tile of `tile` bytes (kmaj_off inverted): its row and first lane
__device__ __forceinline__ void kmaj_at(long long q, int tile, long long* blk, int* row, int* lane) {
  const long long byte = 4 * q;
  *blk = byte / tile;
  const int w = (int)(byte % tile), b = w >> 7, rem = w & 127;
  *row = (b >> 1) * 8 + (rem >> 4);
  *lane = (b & 1) * 16 + (rem & 15);
}

// A' as A's producer writes it (layout.h's lane_a with no coins): row i < m's data from x [m, K], zero data from m on; the
// tags in step 0's lanes 24-27 and the lane-coin A' (ROT_W by row parity) in every step's lanes 28-30, on every computed row;
// the noise lanes 0. A thread a u32 of A', rows tiles of S steps.
__global__ void a_k(uint8_t* __restrict__ A, const int8_t* __restrict__ x, int m, int rows, int K, int S) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)rows * S * (KB / 4)) return;
  long long blk;
  int r, l0;
  kmaj_at(q, TA, &blk, &r, &l0);
  const int i = (int)(blk / S) * 64 + r, t = (int)(blk % S);
  u32 v = 0;
  for (int b = 0; b < 4; b++) {
    const int l = l0 + b, k = kidx_d(t, l, K);
    int a = k >= 0 && i < m ? x[(size_t)i * K + k] : 0;
    if (t == 0 && l >= 24 && l < 28) a = tag_of(i, l - 24);
    if (l >= 28 && l < 31) a = l - 28 == (i & 1) ? 127 : 1;
    v |= (u32)(uint8_t)(int8_t)a << (8 * b);
  }
  reinterpret_cast<u32*>(A)[q] = v;
}
// B' as formed at load (layout.h's lane_b with G 0 and no lane-coin word): column j < N's data from w [N, K], zero data
// from N on; step 0's lanes 21-23 and 31 the code C[0..3, j], a later step's lane 31 1; G's lanes 0 until g_k
__global__ void b_k(uint8_t* __restrict__ B, const int8_t* __restrict__ w, int N, int n, int K, int S) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)n * S * (KB / 4)) return;
  long long blk;
  int c, l0;
  kmaj_at(q, BN * KB, &blk, &c, &l0);
  const int j = (int)(blk / S) * BN + c, t = (int)(blk % S);
  u32 v = 0;
  for (int b = 0; b < 4; b++) {
    const int l = l0 + b, k = kidx_d(t, l, K);
    int a = k >= 0 && j < N ? w[(size_t)j * K + k] : 0;
    if (t == 0 && ((l >= 21 && l < 24) || l == 31)) a = code_of(j, l == 31 ? 3 : l - 21);
    if (t > 0 && l == 31) a = 1;
    v |= (u32)(uint8_t)(int8_t)a << (8 * b);
  }
  reinterpret_cast<u32*>(B)[q] = v;
}
__global__ void xj_k(FoldCol* __restrict__ xj, const float* __restrict__ s, int N, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  xj[j] = FoldCol{j < N ? s[j] : 0.0f, code_word(j)};
}

struct Linear {
  int N = 0, K = 0, S = 0, chunk = 0;
  uint8_t* B = nullptr;
  int* beta = nullptr;
  FoldCol* xj = nullptr;
};
struct State {
  int max_m = 0, nbuf = 0, mode = 2;
  size_t budget = 0;
  bool ready = false;
  std::vector<Linear> lins;
  uint8_t* A = nullptr;
  u32 *head = nullptr, *lanes = nullptr;
  int* src = nullptr;
  FoldRow* xr = nullptr;
  float* xs = nullptr;
  std::vector<uint4*> W;
  std::vector<Hasher> hw;
  Hasher ha;
  std::vector<cudaStream_t> s2;
  std::vector<cudaEvent_t> full, done;
  uint8_t *roots = nullptr, *hroot = nullptr;
  int nroots = 0;
  long long chunks = 0;
};
static State g;

// the root of Hasher::tree_of(n)'s tree (rb > 0): the same walk over its levels, launching nothing
static const uint8_t* tree_top(const Hasher& h, u64 n) {
  u64 w = 1;
  while (w < n) w <<= 1;
  const uint8_t* in = h.leaves;
  uint8_t* out = h.lv;
  while (w > 1) {
    if (w > (1u << 16)) {
      in = out;
      out += 64 * (w / 2);
      w /= 2;
      continue;
    }
    int k = 0;
    while (k < 10 && (1ull << k) < w) k++;
    in = out + 64 * (w - 2 * (w >> k));
    out += 64 * (w - (w >> k));
    w >>= k;
  }
  return in;
}
static unsigned blocks_of(long long threads) { return (unsigned)((threads + 255) / 256); }
static int rows_of(int m) { return (m + 63) / 64 * 64; }

}  // namespace lead

using namespace lead;

extern "C" {

// The lead's layout, calls of at most max_m rows, words chunks of at most budget_gb GB, nbuf hash buffers; 2 if the
// device isn't sm_90
int ssn_lead_init(int max_m, double budget_gb, int nbuf, int mode) {
  int dev = 0;
  cudaDeviceProp p;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&p, dev));
  if (p.major != 9 || max_m < 1 || nbuf < 1 || mode < 0 || mode > 2) return 2;
  g_sms = p.multiProcessorCount;
  g_tag = true, g_blc = 2, g_rot = true, g_fold = true, g_xu = false;
  g.max_m = rows_of(max_m), g.budget = (size_t)(budget_gb * 1e9), g.nbuf = nbuf, g.mode = mode;
  return 0;
}

// Linear w's operands from its weight w8 [N, K] int8 and scale sw [N] f32 (device, row-major); 3 for a shape the lead's
// library doesn't serve (N a whole number of 256-column tiles, K whole KiB), 4 after ready
int ssn_lead_load(int w, int N, int K, const int8_t* w8, const float* sw, cudaStream_t st) {
  if (g.ready) return 4;
  if (w < 0 || N < BN || N % BN || K < A_ROW || K % A_ROW) return 3;
  if ((int)g.lins.size() <= w) g.lins.resize(w + 1);
  Linear& d = g.lins[w];
  d.N = N, d.K = K, d.S = steps_of(K);
  const size_t row_bytes = (size_t)N * d.S * 4;
  d.chunk = (int)std::min<size_t>(g.max_m, std::max<size_t>(64, g.budget / row_bytes / 64 * 64));
  CK(cudaMalloc(&d.B, (size_t)N * d.S * KB));
  b_k<<<blocks_of((long long)N * d.S * (KB / 4)), 256, 0, st>>>(d.B, w8, N, N, K, d.S);
  std::vector<int> beta(N);
  for (int j = 0; j < N; j++) beta[j] = beta_of(j, d.S);
  CK(cudaMalloc(&d.beta, N * sizeof(int)));
  CK(cudaMemcpyAsync(d.beta, beta.data(), N * sizeof(int), cudaMemcpyHostToDevice, st));
  CK(cudaMalloc(&d.xj, N * sizeof(FoldCol)));
  xj_k<<<blocks_of(N), 256, 0, st>>>(d.xj, sw, N, N);
  CK(cudaGetLastError());
  CK(cudaStreamSynchronize(st));
  return 0;
}

// The call buffers for the loaded linears and the hash streams; the device bytes it holds into *bytes
int ssn_lead_ready(unsigned long long* bytes) {
  if (g.ready) return 0;
  if (g.lins.empty()) return 3;
  size_t a = 0, lanes = 0, src = 0, head = 0, words = 0, ab = 0;
  const int M = g.max_m, T = M / 64;
  for (const Linear& d : g.lins) {
    if (!d.B) return 3;
    a = std::max(a, (size_t)M * d.S * KB);
    lanes = std::max(lanes, (size_t)T * d.S * d.N);
    src = std::max(src, (size_t)2 * T * d.N);
    head = std::max(head, (size_t)M * coin_words(d.S) + d.N + coin_words(d.S));
    words = std::max(words, (size_t)d.chunk * d.N * d.S);
    ab = std::max(ab, (size_t)M * d.K);
  }
  size_t held = a + 4 * (lanes + src + head + (size_t)M) + sizeof(FoldRow) * M;
  CK(cudaMalloc(&g.A, a));
  CK(cudaMalloc(&g.lanes, lanes * 4));
  CK(cudaMalloc(&g.src, src * 4));
  CK(cudaMalloc(&g.head, head * 4));
  CK(cudaMalloc(&g.xs, (size_t)M * 4));
  CK(cudaMalloc(&g.xr, sizeof(FoldRow) * M));
  CK(cudaMalloc(&g.roots, (size_t)MAX_CHUNKS * 64));
  CK(cudaMallocHost(&g.hroot, 64));
  g.ha.init(ab, A_ROW, (u64)M);
  if (g.mode > 0) {
    const int nb = g.mode == 2 ? g.nbuf : 1;
    g.W.resize(nb);
    for (auto& w : g.W) CK(cudaMalloc(&w, words * 4 + 2 * 65536));
    held += nb * (words * 4 + 2 * 65536);
  }
  if (g.mode == 2) {
    int least = 0, greatest = 0;
    CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    g.hw.resize(g.nbuf);
    g.s2.resize(g.nbuf), g.full.resize(g.nbuf), g.done.resize(g.nbuf);
    for (int b = 0; b < g.nbuf; b++) {
      g.hw[b].init(words * 4, W_ROW);
      CK(cudaStreamCreateWithPriority(&g.s2[b], cudaStreamNonBlocking, least));
      CK(cudaEventCreateWithFlags(&g.full[b], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&g.done[b], cudaEventDisableTiming));
      CK(cudaEventRecord(g.done[b], g.s2[b]));
    }
  }
  CK(cudaDeviceSynchronize());
  g.ready = true;
  if (bytes) *bytes = held;
  return 0;
}

// The stand-in root(A_j) of linear w's call on x [m, K] (device) into out (64 bytes, host), st synchronized
int ssn_lead_a_root(int w, int m, const int8_t* x, uint8_t* out, cudaStream_t st) {
  if (!g.ready || w < 0 || w >= (int)g.lins.size() || m < 1 || rows_of(m) > g.max_m) return 1;
  const Linear& d = g.lins[w];
  const u64 n = g.ha.leaves_of((const uint8_t*)x, (u64)m, (u64)d.K, st);
  g.ha.tree_of(n, st);
  CK(cudaMemcpyAsync(g.hroot, tree_top(g.ha, n), 64, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  memcpy(out, g.hroot, 64);
  return 0;
}

// Linear w's call on x [m, K] and its scales sa [m] (device) under the feed (host: rows R H words, g C words, lanes
// T S N words, src 2 T N words), serving y [64 ceil(m / 64), N] bf16 (device) on st
int ssn_lead_call(int w, int m, const int8_t* x, const float* sa, const u32* rows, const u32* gw, const u32* lanes,
                  const int* src, void* y, cudaStream_t st) {
  if (!g.ready || w < 0 || w >= (int)g.lins.size() || m < 1 || rows_of(m) > g.max_m) return 1;
  const Linear& d = g.lins[w];
  const int M = rows_of(m), T = M / 64, R = pad_rows(m), C = pad_cols(d.N), H = coin_words(d.S), n = d.N, S = d.S;
  CK(cudaMemcpyAsync(g.head, rows, (size_t)R * H * 4, cudaMemcpyHostToDevice, st));
  CK(cudaMemcpyAsync(g.head + (size_t)R * H, gw, (size_t)C * 4, cudaMemcpyHostToDevice, st));
  CK(cudaMemsetAsync(g.head + (size_t)R * H + C, 0, (size_t)H * 4, st));
  CK(cudaMemcpyAsync(g.lanes, lanes, (size_t)T * S * n * 4, cudaMemcpyHostToDevice, st));
  CK(cudaMemcpyAsync(g.src, src, (size_t)2 * T * n * 4, cudaMemcpyHostToDevice, st));
  CK(cudaMemcpyAsync(g.xs, sa, (size_t)m * 4, cudaMemcpyDeviceToDevice, st));
  if (M > m) CK(cudaMemsetAsync(g.xs + m, 0, (size_t)(M - m) * 4, st));
  a_k<<<blocks_of((long long)M * S * (KB / 4)), 256, 0, st>>>(g.A, x, m, M, d.K, S);
  CK(cudaGetLastError());
  form_g(g.head, g.A, d.B, M, n, R, C, S, BN, st, FORM | GCOPY, 0, g.xr);
  for (int r0 = 0; r0 < M; r0 += d.chunk) {
    const int mc = std::min(d.chunk, M - r0), b = g.mode == 2 ? (int)(g.chunks % g.nbuf) : 0;
    if (g.mode == 2) CK(cudaStreamWaitEvent(st, g.done[b]));
    const Xu xu{g.xr + r0, g.xs + r0, g.src + (size_t)(r0 / 64) * 2 * n, d.xj, (u32*)y + (size_t)r0 * n / 2, 2};
    gemm(BN, g.mode > 0, g.A + (size_t)(r0 / 64) * S * TA, d.B, d.beta, mc, n, S, g.mode > 0 ? g.W[b] : nullptr, nullptr,
         nullptr, st, g.lanes + (size_t)(r0 / 64) * S * n, xu);
    if (g.mode == 2) {
      if (g.nroots >= MAX_CHUNKS) return 5;
      CK(cudaEventRecord(g.full[b], st));
      CK(cudaStreamWaitEvent(g.s2[b], g.full[b]));
      const u64 nl = g.hw[b].leaves_of((const uint8_t*)g.W[b], (u64)(mc / 64) * (n / BN) * S, (u64)64 * BN * 4, g.s2[b]);
      g.hw[b].tree_of(nl, g.s2[b]);
      CK(cudaMemcpyAsync(g.roots + 64 * (size_t)g.nroots++, tree_top(g.hw[b], nl), 64, cudaMemcpyDeviceToDevice, g.s2[b]));
      CK(cudaEventRecord(g.done[b], g.s2[b]));
    }
    g.chunks++;
  }
  return 0;
}

// The hash streams drained and the chunks' roots since the last call (64 bytes each, at most cap) into out (host): their
// count, or -1 past cap
int ssn_lead_words_roots(uint8_t* out, int cap) {
  for (cudaStream_t s : g.s2) CK(cudaStreamSynchronize(s));
  const int n = g.nroots;
  if (n > cap) return -1;
  if (n) {
    cudaStream_t s = g.s2.empty() ? nullptr : g.s2[0];
    CK(cudaMemcpyAsync(out, g.roots, (size_t)n * 64, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
  }
  g.nroots = 0;
  return n;
}

// a_k and b_k against ssn.cu's tile_a and tile_b on random rows and weight (m rows, N columns, K): the bytes that differ
int ssn_lead_check_operands(int m, int N, int K, unsigned long long seed) {
  if (N < BN || N % BN || m < 1) return -1;
  const int M = rows_of(m);
  ::Lin L{M, N, K, steps_of(K), BN, {}, {}};
  fill(L, 0, seed);
  std::fill(L.A.begin() + (size_t)m * K, L.A.end(), (int8_t)0);
  const std::vector<uint8_t> ta = tile_a(L), tb = tile_b(L);
  int8_t* dx = up(L.A);
  int8_t* dw = up(L.B);
  uint8_t *dA, *dB;
  CK(cudaMalloc(&dA, ta.size()));
  CK(cudaMalloc(&dB, tb.size()));
  CK(cudaMemset(dA, 0xA5, ta.size()));
  CK(cudaMemset(dB, 0xA5, tb.size()));
  a_k<<<blocks_of((long long)M * L.S * (KB / 4)), 256>>>(dA, dx, m, M, K, L.S);
  b_k<<<blocks_of((long long)N * L.S * (KB / 4)), 256>>>(dB, dw, N, N, K, L.S);
  CK(cudaGetLastError());
  std::vector<uint8_t> ga(ta.size()), gb(tb.size());
  CK(cudaMemcpy(ga.data(), dA, ga.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gb.data(), dB, gb.size(), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (size_t i = 0; i < ta.size(); i++) bad += ga[i] != ta[i];
  for (size_t i = 0; i < tb.size(); i++) bad += gb[i] != tb[i];
  CK(cudaFree(dx));
  CK(cudaFree(dw));
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  return bad;
}

}  // extern "C"
