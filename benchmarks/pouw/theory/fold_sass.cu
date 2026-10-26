// The served receipt folds as per-lane word chains, for the SASS instruction mix FOLD-OPT prices on each device: fx1 on
// sm_120, li on the H200 (sm_90a) and li heavy (li + keyed rounds) on the B200 (sm_100a), and li heavy with its keyed
// rounds' constant XOR on the IMAD pipe (`fold_min.py`): key-oblivious (61 IMAD-pipe instructions a round) and at li's
// cheapest key LK[6][0] (25). Each kernel is the reference fold of `verity/ml/pouw/circuit/segment.py`
// (`_fx1_absorb`, `_li_absorb`, and the keyed round `y <- (y ^ k) * a + c`), one chain a lane. The absorbed words are
// per-lane registers (loaded before the clock starts); keys and multipliers are in constant memory, uniform as in the
// served kernels, so ptxas can neither specialise a multiplier into a shift nor move a word's work to the uniform
// datapath. Built for sm_90a, sm_100a and sm_120a (`fold_sass.sh`); `fold_sass_count.py` counts each loop body's SASS
// by census class. `main` runs on the device it finds:
//   - bitexact: one warp, one pass, every lane recomputed on the host from the same steps;
//   - lat: that warp's clock64 a word (the dependent chain's latency, not a certificate's price);
//   - tp: SMSP-cycles a warp-word with every SM full (one wave of the occupancy's blocks), clock64 a block.
#include <stdint.h>
#include <stdio.h>

#define L 32       // fx1: words a pass; li: pairs a pass (2L words)
#define NT 256     // threads a block

__constant__ uint32_t KM[2 * L];   // fx1: the key multiplier m; li: a; keyed round r (at L + r): a
__constant__ uint32_t KX[2 * L];   // xor keys
__constant__ uint32_t KC[2 * L];   // fx1: y's multiplier a; keyed round: c
__constant__ uint32_t SH[31];      // SH[b] = 2^(32 - b)
__constant__ uint32_t A0[L], C0[L], CF[L][31];      // key-oblivious IMAD route of keyed round r (random keys)
__constant__ uint32_t A0S[L], C0S[L], CFS[L][31];   // the IMAD route at KSPEC

#define KSPEC 0x9e13b91cu  // segment.LK[6][0]: 12 bit changes in 1..30, so 25 IMAD-pipe instructions a round
#define ROT(j) ((((j) * 7 + 5) % 31) + 1)
#define KEYED(j) (((j) & 3) != 0)  // 3 of every 4 pairs: 3/8 of a round a word, as li heavy folds

__device__ __forceinline__ uint32_t rotl(uint32_t x, int r) { return __funnelshift_l(x, x, r); }

__device__ __forceinline__ uint32_t keyed_imad_all(uint32_t y, int r) {
  uint32_t acc = y * A0[r] + C0[r];
#pragma unroll
  for (int b = 1; b <= 30; ++b) acc = __umulhi(y, SH[b]) * CF[r][b] + acc;
  return acc;
}

__device__ __forceinline__ uint32_t keyed_imad_spec(uint32_t y, int r) {
  uint32_t acc = y * A0S[r] + C0S[r];
#pragma unroll
  for (int b = 1; b <= 30; ++b)
    if (((KSPEC >> b) ^ (KSPEC >> (b - 1))) & 1u) acc = __umulhi(y, SH[b]) * CFS[r][b] + acc;
  return acc;
}

#define PROLOGUE(NW)                                                    \
  const int idx = blockIdx.x * blockDim.x + threadIdx.x, ln = idx & 31; \
  uint32_t w[NW];                                                       \
  _Pragma("unroll") for (int j = 0; j < NW; ++j) w[j] = W[j * 32 + ln]; \
  uint32_t y = st[ln] ^ (uint32_t)idx * 0x9E3779B9u;                    \
  __syncthreads();                                                      \
  long long t0 = clock64();

#define EPILOGUE                                     \
  __syncthreads();                                   \
  long long t1 = clock64();                          \
  out[idx] = y;                                      \
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;

extern "C" __global__ void fx1_chain(const uint32_t *st, const uint32_t *W, uint32_t *out, long long *cyc, int reps) {
  PROLOGUE(L)
  uint32_t z = st[32 + ln];
#pragma unroll 1
  for (int i = 0; i < reps; ++i) {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      uint32_t u = y ^ (w[j] * KM[j]) ^ KX[j];
      y = u * KC[j] + z;
      z = rotl(u, ROT(j)) + y;
    }
  }
  y ^= rotl(z, 16);
  EPILOGUE
}

extern "C" __global__ void li_chain(const uint32_t *st, const uint32_t *W, uint32_t *out, long long *cyc, int reps) {
  PROLOGUE(2 * L)
#pragma unroll 1
  for (int i = 0; i < reps; ++i) {
#pragma unroll
    for (int j = 0; j < L; ++j) y = (y ^ w[2 * j] ^ KX[j]) * KM[j] + w[2 * j + 1];
  }
  EPILOGUE
}

extern "C" __global__ void li_heavy_chain(const uint32_t *st, const uint32_t *W, uint32_t *out, long long *cyc,
                                          int reps) {
  PROLOGUE(2 * L)
#pragma unroll 1
  for (int i = 0; i < reps; ++i) {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      y = (y ^ w[2 * j] ^ KX[j]) * KM[j] + w[2 * j + 1];
      if (KEYED(j)) y = (y ^ KX[L + j]) * KM[L + j] + KC[L + j];
    }
  }
  EPILOGUE
}

extern "C" __global__ void li_heavy_imad_chain(const uint32_t *st, const uint32_t *W, uint32_t *out, long long *cyc,
                                               int reps) {
  PROLOGUE(2 * L)
#pragma unroll 1
  for (int i = 0; i < reps; ++i) {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      y = (y ^ w[2 * j] ^ KX[j]) * KM[j] + w[2 * j + 1];
      if (KEYED(j)) y = keyed_imad_all(y, j);
    }
  }
  EPILOGUE
}

extern "C" __global__ void li_heavy_imad25_chain(const uint32_t *st, const uint32_t *W, uint32_t *out, long long *cyc,
                                                 int reps) {
  PROLOGUE(2 * L)
#pragma unroll 1
  for (int i = 0; i < reps; ++i) {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      y = (y ^ w[2 * j] ^ KX[j]) * KM[j] + w[2 * j + 1];
      if (KEYED(j)) y = keyed_imad_spec(y, j);
    }
  }
  EPILOGUE
}

#ifndef __CUDA_ARCH__
#include <cstdlib>
#include <cuda_runtime.h>

typedef void (*Kern)(const uint32_t *, const uint32_t *, uint32_t *, long long *, int);

static uint32_t h_rotl(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }
static uint32_t h_keyed(uint32_t y, uint32_t k, uint32_t a, uint32_t c) { return (y ^ k) * a + c; }

static uint32_t km[2 * L], kx[2 * L], kc[2 * L], st[64], W[2 * L * 32];

// the reference fold of lane ln's chain over one pass; kind 0 fx1, 1 li, 2 li heavy, 3 li heavy at KSPEC
static uint32_t ref(int kind, int ln) {
  uint32_t y = st[ln] ^ (uint32_t)ln * 0x9E3779B9u;
  if (kind == 0) {
    uint32_t z = st[32 + ln];
    for (int j = 0; j < L; ++j) {
      uint32_t u = y ^ (W[j * 32 + ln] * km[j]) ^ kx[j];
      y = u * kc[j] + z;
      z = h_rotl(u, ((j * 7 + 5) % 31) + 1) + y;
    }
    return y ^ h_rotl(z, 16);
  }
  for (int j = 0; j < L; ++j) {
    y = (y ^ W[2 * j * 32 + ln] ^ kx[j]) * km[j] + W[(2 * j + 1) * 32 + ln];
    if (kind >= 2 && (j & 3) != 0) y = h_keyed(y, kind == 3 ? KSPEC : kx[L + j], km[L + j], kc[L + j]);
  }
  return y;
}

static void imad_route(uint32_t K, uint32_t a, uint32_t c, uint32_t *a0, uint32_t *c0, uint32_t *cf) {
  uint32_t k0 = K & 1u;
  *a0 = a * (1u - 2u * k0);
  *c0 = K * a + c;
  cf[0] = 0;
  for (int b = 1; b <= 30; ++b) {
    int change = ((K >> b) ^ (K >> (b - 1))) & 1u;
    uint32_t tau = change ? (((K >> b) & 1u) ? 0xFFFFFFFFu : 1u) : 0u;
    cf[b] = tau * a * (1u << (b + 1));
  }
}

int main() {
  int dev = 0, sms = 0, maj = 0, mnr = 0, clk = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev);
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
  printf("device cc=%d.%d sms=%d clock_khz=%d\n", maj, mnr, sms, clk);

  srand(0xF01D);
  for (int i = 0; i < 2 * L; ++i) {
    km[i] = ((uint32_t)rand() * 40503u) | 1u;
    kx[i] = (uint32_t)rand() * 2246822519u;
    kc[i] = (uint32_t)rand() * 3266489917u;
  }
  for (int i = 0; i < 64; ++i) st[i] = (uint32_t)rand() * 668265263u;
  for (int i = 0; i < 2 * L * 32; ++i) W[i] = (uint32_t)rand() * 2654435761u;
  uint32_t sh[31], a0[L], c0[L], cf[L][31], a0s[L], c0s[L], cfs[L][31];
  sh[0] = 0;
  for (int b = 1; b <= 30; ++b) sh[b] = 1u << (32 - b);
  for (int r = 0; r < L; ++r) {
    imad_route(kx[L + r], km[L + r], kc[L + r], &a0[r], &c0[r], cf[r]);
    imad_route(KSPEC, km[L + r], kc[L + r], &a0s[r], &c0s[r], cfs[r]);
  }
  cudaMemcpyToSymbol(KM, km, sizeof km);
  cudaMemcpyToSymbol(KX, kx, sizeof kx);
  cudaMemcpyToSymbol(KC, kc, sizeof kc);
  cudaMemcpyToSymbol(SH, sh, sizeof sh);
  cudaMemcpyToSymbol(A0, a0, sizeof a0);
  cudaMemcpyToSymbol(C0, c0, sizeof c0);
  cudaMemcpyToSymbol(CF, cf, sizeof cf);
  cudaMemcpyToSymbol(A0S, a0s, sizeof a0s);
  cudaMemcpyToSymbol(C0S, c0s, sizeof c0s);
  cudaMemcpyToSymbol(CFS, cfs, sizeof cfs);

  struct { const char *name; Kern k; int kind, words; } ks[] = {
      {"fx1_chain", fx1_chain, 0, L},
      {"li_chain", li_chain, 1, 2 * L},
      {"li_heavy_chain", li_heavy_chain, 2, 2 * L},
      {"li_heavy_imad_chain", li_heavy_imad_chain, 2, 2 * L},
      {"li_heavy_imad25_chain", li_heavy_imad25_chain, 3, 2 * L},
  };
  const int maxblocks = sms * 32;
  uint32_t *d_st, *d_W, *d_out;
  long long *d_cyc;
  cudaMalloc(&d_st, sizeof st);
  cudaMalloc(&d_W, sizeof W);
  cudaMalloc(&d_out, (size_t)maxblocks * NT * sizeof(uint32_t));
  cudaMalloc(&d_cyc, maxblocks * sizeof(long long));
  cudaMemcpy(d_st, st, sizeof st, cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, W, sizeof W, cudaMemcpyHostToDevice);
  long long *cyc = (long long *)malloc(maxblocks * sizeof(long long));
  int all = 1;
  const int REPS = 512;
  for (auto &e : ks) {
    uint32_t out[32];
    e.k<<<1, 32>>>(d_st, d_W, d_out, d_cyc, 1);
    cudaError_t err = cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost);
    int bad = err != cudaSuccess;
    for (int ln = 0; ln < 32; ++ln) bad += out[ln] != ref(e.kind, ln);
    all &= bad == 0;
    e.k<<<1, 32>>>(d_st, d_W, d_out, d_cyc, REPS);
    cudaDeviceSynchronize();
    cudaMemcpy(cyc, d_cyc, sizeof(long long), cudaMemcpyDeviceToHost);
    double lat = (double)cyc[0] / ((double)e.words * REPS);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, e.k, NT, 0);
    if (occ > 32) occ = 32;
    int blocks = sms * occ;
    e.k<<<blocks, NT>>>(d_st, d_W, d_out, d_cyc, REPS);
    err = cudaDeviceSynchronize();
    cudaMemcpy(cyc, d_cyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
    long long mx = 0;
    for (int b = 0; b < blocks; ++b) mx = cyc[b] > mx ? cyc[b] : mx;
    double warps_smsp = occ * (NT / 32) / 4.0;
    double tp = (double)mx / (warps_smsp * e.words * REPS);
    printf("%s bitexact=%d lat_cyc_per_word=%.4f tp_smsp_cyc_per_warp_word=%.5f occ_blocks=%d warps_per_smsp=%.1f "
           "err=%s\n", e.name, bad == 0, lat, tp, occ, warps_smsp, cudaGetErrorString(err));
  }
  printf("all_bitexact=%d\n", all);
  return all ? 0 : 3;
}
#endif
