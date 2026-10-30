// The lead's kernel (ssn.cu under BLC=2, ROT=1, FOLD=1: blc4a_t64_e7_s1_rot_fold) as a library a serving process calls
// on its own device tensors; `serve_lead.py` binds it as the lead's `LeadBackend`.
//   load      a linear's B' from its int8 weight [N, K] (N whole 256-column tiles), beta_j and its column words
//             (layout.h's FoldCol: s_j from its weight scale and the code C[0..3, j]), formed on the device once
//   ready     the call's buffers, sized from the loaded linears
//   call      A' formed on the device from the call's s8 rows (a_k, tile_a's layout), the certifier's feed copied in
//             (row words, G words and H zero words; lane words and column sources beta_j - sigma at stride N), the
//             forming, which also writes each row's fold words (layout.h's FoldRow), and G's copy (form_g), then the GEMM,
//             the fold serving verity_int8's y bf16 [64 ceil(m / 64), N]. Given a words buffer it exports every computed
//             word there in ssn.cu's export layout (64 ceil(m / 64) N S words), which the caller commits; without one the
//             words stay in registers (the GEMM serving y alone). It hashes nothing.
//   check_operands  a_k and b_k against ssn.cu's host tile_a and tile_b
// Build: served_lead.sh (with native_rows.cu and its plain-leaf copy, as run.sh builds ssn).
#define main ssn_main
#include "ssn.cu"
#undef main

namespace lead {

constexpr int BN = 256;

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
  int N = 0, K = 0, S = 0;
  uint8_t* B = nullptr;
  int* beta = nullptr;
  FoldCol* xj = nullptr;
};
struct State {
  int max_m = 0;
  bool ready = false;
  std::vector<Linear> lins;
  uint8_t* A = nullptr;
  u32 *head = nullptr, *lanes = nullptr;
  int* src = nullptr;
  FoldRow* xr = nullptr;
  float* xs = nullptr;
};
static State g;

static unsigned blocks_of(long long threads) { return (unsigned)((threads + 255) / 256); }
static int rows_of(int m) { return (m + 63) / 64 * 64; }

}  // namespace lead

using namespace lead;

extern "C" {

// The lead's layout, calls of at most max_m rows; 2 if the device isn't sm_90
int ssn_lead_init(int max_m) {
  int dev = 0;
  cudaDeviceProp p;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&p, dev));
  if (p.major != 9 || max_m < 1) return 2;
  g_sms = p.multiProcessorCount;
  g_tag = true, g_blc = 2, g_rot = true, g_fold = true, g_xu = false;
  g.max_m = rows_of(max_m);
  return 0;
}

// Linear w's operands from its weight w8 [N, K] int8 and scale sw [N] f32 (device, row-major); 3 for a shape the lead's
// library doesn't serve (N a whole number of 256-column tiles), 4 after ready
int ssn_lead_load(int w, int N, int K, const int8_t* w8, const float* sw, cudaStream_t st) {
  if (g.ready) return 4;
  if (w < 0 || N < BN || N % BN || K < 1) return 3;
  if ((int)g.lins.size() <= w) g.lins.resize(w + 1);
  Linear& d = g.lins[w];
  d.N = N, d.K = K, d.S = steps_of(K);
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

// The call buffers for the loaded linears; the device bytes it holds into *bytes
int ssn_lead_ready(unsigned long long* bytes) {
  if (g.ready) return 0;
  if (g.lins.empty()) return 3;
  size_t a = 0, lanes = 0, src = 0, head = 0;
  const int M = g.max_m, T = M / 64;
  for (const Linear& d : g.lins) {
    if (!d.B) return 3;
    a = std::max(a, (size_t)M * d.S * KB);
    lanes = std::max(lanes, (size_t)T * d.S * d.N);
    src = std::max(src, (size_t)2 * T * d.N);
    head = std::max(head, (size_t)M * coin_words(d.S) + d.N + coin_words(d.S));
  }
  const size_t held = a + 4 * (lanes + src + head + (size_t)M) + sizeof(FoldRow) * M;
  CK(cudaMalloc(&g.A, a));
  CK(cudaMalloc(&g.lanes, lanes * 4));
  CK(cudaMalloc(&g.src, src * 4));
  CK(cudaMalloc(&g.head, head * 4));
  CK(cudaMalloc(&g.xs, (size_t)M * 4));
  CK(cudaMalloc(&g.xr, sizeof(FoldRow) * M));
  CK(cudaDeviceSynchronize());
  g.ready = true;
  if (bytes) *bytes = held;
  return 0;
}

// Linear w's call on x [m, K] and its scales sa [m] (device) under the feed (host: rows R H words, g C words, lanes
// T S N words, src 2 T N words), serving y [64 ceil(m / 64), N] bf16 (device) on st, and exporting its words into
// words (device, 64 ceil(m / 64) N S of them) unless it is null
int ssn_lead_call(int w, int m, const int8_t* x, const float* sa, const u32* rows, const u32* gw, const u32* lanes,
                  const int* src, void* y, u32* words, cudaStream_t st) {
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
  const Xu xu{g.xr, g.xs, g.src, d.xj, (u32*)y, 2};
  gemm(BN, words != nullptr, g.A, d.B, d.beta, M, n, S, (uint4*)words, nullptr, nullptr, st, g.lanes, xu);
  return 0;
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
